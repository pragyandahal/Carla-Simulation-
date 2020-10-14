# This is a program made by Pragyan for Python Exercise and Perception analysis
import glob
import os
import sys
import numpy as np
import cv2
import math
import open3d as o3d
from matplotlib import pyplot as plt
import matlab.engine
eng = matlab.engine.start_matlab()


import matplotlib.image as mpimg
import numpy as np
#%matplotlib inline

#Loading Road Information

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
import time
import random

try:
    import pygame
except ImportError:
    print("Come on , you need to install pygame")

try:
    import queue
except ImportError:
    import Queue as queue


im_w = 640;
im_h = 480;
check = False;

class CarlaSyncMode(object):
    """
    Context manager to synchronize output from different sensors. Synchronous
    mode is enabled as long as we are inside this context
        with CarlaSyncMode(world, sensors) as sync_mode:
            while True:
                data = sync_mode.tick(timeout=1.0)
    """

    def __init__(self, world, *sensors, **kwargs):
        self.world = world
        self.sensors = sensors
        self.frame = None
        self.delta_seconds = 1.0 / kwargs.get('fps', 20)
        self._queues = []
        self._settings = None

    def __enter__(self):
        self._settings = self.world.get_settings()
        self.frame = self.world.apply_settings(carla.WorldSettings(
            no_rendering_mode=False,
            synchronous_mode=True,
            fixed_delta_seconds=self.delta_seconds))

        def make_queue(register_event):
            q = queue.Queue()
            register_event(q.put)
            self._queues.append(q)

        make_queue(self.world.on_tick)
        for sensor in self.sensors:
            make_queue(sensor.listen)
        return self

    def tick(self, timeout):
        self.frame = self.world.tick()
        data = [self._retrieve_data(q, timeout) for q in self._queues]
        assert all(x.frame == self.frame for x in data)
        return data

    def __exit__(self, *args, **kwargs):
        self.world.apply_settings(self._settings)

    def _retrieve_data(self, sensor_queue, timeout):
        while True:
            data = sensor_queue.get(timeout=timeout)
            if data.frame == self.frame:
                return data


class ImageProcessing(object):

    def __init__(self):
        self.low_threshold = 50
        self.high_threshold = 150
        self.kernel_size = 5
        self.rho = 2 # distance resolution in pixels of the Hough grid
        self.theta = np.pi/180 # angular resolution in radians of the Hough grid
        self.threshold = 100   # minimum number of votes (intersections in Hough grid cell)
        self.min_line_length = 10 #minimum number of pixels making up a line
        self.max_line_gap = 20    # maximum gap in pixels between connectable line segments
        self.im_width = im_w
        self.im_height = im_h

    def grayscale(self,img):
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # Or use BGR2GRAY if you read an image with cv2.imread()
        # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def canny(self,img):
        """Applies the Canny transform"""
        return cv2.Canny(img, self.low_threshold, self.high_threshold)

    def gaussian_blur(self,img):
        """Applies a Gaussian Noise kernel"""
        return cv2.GaussianBlur(img, (self.kernel_size, self.kernel_size), 0)

    def make_points(self,image,line):
        """
        It computes the end points of the lines from the values of slopeand intercept. The y values are at the bottom and at 3/5 of the
        image, that is a value chosen as the end of the near section of
        the image.
        """
        print("This is line inside make_points: ",line)
        try:
            slope, intercept = line
            y1 = int(image.shape[0]) # bottom of the image
            y2 = int(y1*3/5) # slightly lower than the middle
            x1 = int((y1 - intercept)/slope)
            x2 = int((y2 - intercept)/slope)
            return [[x1, y1, x2, y2]]
        except:
             return None

    def average_slope_intercept(self,image):
        """
        1st order fitting of the line's points and separation in left
        line and right line based on the intercept sign. Then it averages
        the slope and intercept values of both the lines and computes the
        start and end points for both.
        """
        left_fit = []
        right_fit = []
        if self.lines is None:
            return None
        for line in self.lines:
            for x1, y1, x2, y2 in line:
                # Polyfit computes the 1st order fitting of the lane points
                fit = np.polyfit((x1,x2), (y1,y2), 1)
                slope = fit[0]
                intercept = fit[1]
                if slope < 0: # y is reversed in image
                    left_fit.append((slope, intercept))
                else:
                    right_fit.append((slope, intercept))
        # add more weight to longer lines
        left_fit_average = np.average(left_fit, axis=0)
        right_fit_average = np.average(right_fit, axis=0)
        self.left_line = self.make_points(image,left_fit_average)
        self.right_line = self.make_points(image,right_fit_average)
        self.averaged_lines = [self.left_line, self.right_line]
        return self.averaged_lines

    def region_of_interest(self,img):
        #defining a blank mask to start with
        self.mask = np.zeros_like(img)

        #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
        if len(img.shape) > 2:
            channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
            ignore_mask_color = (255,) * channel_count
        else:
            ignore_mask_color = 255

        #filling pixels inside the polygon defined by "vertices" with the fill color
        cv2.fillPoly(self.mask, self.vertices, ignore_mask_color)
        #cv2.imshow("Mask image after fillPoly",self.mask)

        #returning the image only where mask pixels are nonzero
        self.masked_image = cv2.bitwise_and(img, self.mask)
        #self.masked_image = np.zeros((self.masked_image0.shape[0], self.masked_image0.shape[1], 3), dtype=np.uint8)
        #self.masked_image = np.array([self.masked_image0])
        if check:
            cv2.imshow("Masked image inside region_of_interest",self.masked_image)
        return self.masked_image

    def draw_lines(self,color=[0, 255, 0], thickness=2):
        if self.averaged_lines is not None:
            for line in self.averaged_lines:
                for x1,y1,x2,y2 in line:
                    cv2.line(self.line_img, (x1, y1), (x2, y2), color, thickness)


    def hough_lines(self,img):
        self.lines = cv2.HoughLinesP(img, self.rho, self.theta, self.threshold, np.array([]), minLineLength=self.min_line_length, maxLineGap=self.max_line_gap)
        #self.averaged_lines = self.average_slope_intercept(img)
        self.averaged_lines=self.lines
        self.line_img = np.zeros((self.imshape[0], self.imshape[1], 3), dtype=np.uint8)
        self.draw_lines()
        return self.line_img

    def weighted_img(self):
        α = 0.8
        return cv2.addWeighted(self.line_image, α, self.real_image, 1, 0.)

    def make_image_ready(self,image):
        self.i = np.array(image.raw_data)
        self.i2 = self.i.reshape((self.im_height, self.im_width, 4))
        self.i3 = self.i2[:,:,:3]     # Because in the image, we only need rbg values but not the a at the end
        #cv2.imshow("Camera_Image",self.i3)
        #cv2.waitKey(1)
        return self.i3

    def process_image(self,image):
        # NOTE: The output you return should be a color image (3 channel) for processing video below
        # TODO: put your pipeline here,
        #First, convert the image to greyscale
        self.real_image=image
        self.image_gray =  self.grayscale(self.real_image)

        #Use Gaussian Blur
        self.image_GB = self.gaussian_blur(self.image_gray)

        # Use Canny edge detecters
        self.image_canny = self.canny(self.image_GB)
        #cv2.imshow("Canny Image",self.image_canny)

        # Creating a mask
        self.imshape = self.real_image.shape
        height=self.imshape[0]
        #self.vertices = np.array([[(200, height),(550, 250),(1100, height),]], np.int32)
        self.vertices = np.array([[(self.imshape[1]/4-50,self.imshape[0]),(self.imshape[1]/3+50, self.imshape[0]/2+50), (self.imshape[1]/3+150, self.imshape[0]/2+50), (self.imshape[1]*2/3+150,self.imshape[0])]], dtype=np.int32)
        self.masked_image = self.region_of_interest(self.image_canny)

        # Applying Hough Transformation

        #cv2.imshow("This is masked image",self.masked_image)
        self.line_image = self.hough_lines(self.masked_image)
        self.final_image = self.weighted_img()
        # you should return the final output (image where lines are drawn on lanes)
        self.result = self.final_image
        return self.result

# Definition of the class
class MySimulation(object):
    """docstring for ."""
    def __init__(self,defined_world,vehicle):
        self.actor_list = []
        self.world = defined_world
        debug = self.world.debug
        self.blueprint_library = self.world.get_blueprint_library()
        self.im_width = im_w
        self.im_height = im_h
        self.actor_list = []
        self.ego_vehicle = vehicle

    def depth_Calculation(self,img):
        #self.i4 = img.convert(carla.ColorConverter.Depth)
        #print("This is dir(i4) :", dir(self.i4))
        self.i5 = np.array(img.raw_data)
        self.i6 = self.i5.reshape((self.im_height,self.im_width, 4))
        self.i7 = self.i6[:,:,:3]
        if check:
            cv2.imshow("Depth_Image",self.i7)
        cv2.waitKey(1)
        self.depth_image = self.i7
        R = self.i7[:,:,0]
        G = self.i7[:,:,1]
        B = self.i7[:,:,2]
        normalized = (R + G * 256 + B * 256 * 256) / (256 * 256 * 256 - 1)
        in_meters = 1000 * normalized
        #cv2.imshow('Depth Image in Depth Scale',normalized)
        #cv2.waitKey(1)

    def add_sensors(self,Image_Object):

        #Adding Camera in front of the vehicle
        self.camera_bp = self.blueprint_library.find("sensor.camera.rgb")
        self.camera_bp.set_attribute("image_size_x",f"{self.im_width}")
        self.camera_bp.set_attribute("image_size_y",f"{self.im_height}")
        self.camera_bp.set_attribute("fov","110")
        self.camera_transform = carla.Transform(carla.Location(x=0, z=2.4))
        self.camera = self.world.spawn_actor(self.camera_bp, self.camera_transform, attach_to=self.ego_vehicle,attachment_type=carla.AttachmentType.Rigid)
        self.actor_list.append(self.camera)
        print("Front Camera Created")

        #Adding Camera in rear of the vehicle
        self.camera_transform_rear = carla.Transform(carla.Location(x=-1, z=2.4),carla.Rotation(yaw=180))
        self.camera_rear = self.world.spawn_actor(self.camera_bp, self.camera_transform_rear, attach_to=self.ego_vehicle,attachment_type=carla.AttachmentType.Rigid)
        self.actor_list.append(self.camera_rear)
        print("Rear Camera Created")

        # Adding Depht Camera
        self.depth_bp = self.blueprint_library.find('sensor.camera.depth')
        self.depth_bp.set_attribute("image_size_x",f"{self.im_width}")
        self.depth_bp.set_attribute("image_size_y",f"{self.im_height}")
        self.depth_location = carla.Location(0,0,2)
        self.depth_rotation = carla.Rotation(0,0,0)
        self.depth_transform = carla.Transform(self.depth_location,self.depth_rotation)
        self.depth_Camera = self.world.spawn_actor(self.depth_bp,self.depth_transform,attach_to=self.ego_vehicle,attachment_type=carla.AttachmentType.Rigid)
        self.depth_Camera.listen(lambda data: self.depth_Calculation(data))
        self.actor_list.append(self.depth_Camera)
        print('created %s' % self.depth_Camera.type_id)

        self.lidar_bp = self.blueprint_library.find('sensor.lidar.ray_cast')
        self.lidar_bp.set_attribute('channels',str(64))
        self.lidar_bp.set_attribute('points_per_second',str(90000))
        self.lidar_bp.set_attribute('range',str(20))
        self.lidar_location = carla.Location(0,0,1)
        self.lidar_rotation = carla.Rotation(0,0,0)
        self.lidar_transform = carla.Transform(self.lidar_location,self.lidar_rotation)
        self.lidar_transform = carla.Transform(carla.Location(x=1.5, z=3.5))
        self.lidar = self.world.spawn_actor(self.lidar_bp, self.lidar_transform, attach_to=self.ego_vehicle,attachment_type=carla.AttachmentType.Rigid)
        #self.lidar.listen(lambda point_cloud: self.lidar_processing(point_cloud))

        #self.lidar.listen(lambda point_cloud: point_cloud.save_to_disk('tutorial/new_pcd/%.6d.ply' % point_cloud.frame))
        self.actor_list.append(self.lidar)
        print('created %s' % self.lidar.type_id)

    def Processing(self):
        print("Ego Vehicle Center of Mass :",self.ego_vehicle.bounding_box.location)
        print("Ego Vehicle Extension :",self.ego_vehicle.bounding_box.extent)


class AddingVehicles(object):

    def __init__(self,Simulation_Object,start_pose):
        self.world = Simulation_Object.world
        self.blueprint_library = self.world.get_blueprint_library()
        self.ego_vehicle = Simulation_Object.ego_vehicle
        self.starting_point = start_pose
        self.Simulation_Object = Simulation_Object

    def Vehicle_One(self):
        self.Vehicle1_starting_pose = carla.Transform(self.starting_point.location+carla.Location(x=10),self.starting_point.rotation)
        self.Vehicle1 = self.world.spawn_actor(
            self.blueprint_library.filter("model3")[0],
            self.Vehicle1_starting_pose)
        self.Simulation_Object.actor_list.append(self.Vehicle1)
        self.Vehicle1.set_simulate_physics(False)

    def Vehicle_Two(self):
        self.Vehicle2_starting_pose = carla.Transform(self.starting_point.location+carla.Location(x=-10),self.starting_point.rotation)
        self.Vehicle2 = self.world.spawn_actor(
            self.blueprint_library.filter("Chevrolet")[0],
            self.Vehicle2_starting_pose)
        self.Simulation_Object.actor_list.append(self.Vehicle2)
        self.Vehicle2.set_simulate_physics(False)

class RoadProcessing(object):
    def __init__(self,waypoint):
        self.waypoint_front = waypoint.next_until_lane_end(0.1)
        self.waypoint_rear = waypoint.previous_until_lane_start(0.1)

    def PointCollection_Front(self):
        with open('Front_Waypoints.txt', 'w') as f:
            self.points_front =np.zeros((len(self.waypoint_front),2))
            count_front = 0
            for w in self.waypoint_front:
                point = w.transform.location
                #print("This is point in the transform front: ", point)
                x_cor = point.x
                y_cor = point.y
                count_front = count_front + 1
                #self.points_front[count_front][:] = [x_cor, y_cor]
                f.write("%f %f\n" %(x_cor, y_cor))
                #w_front.append([x_cor,y_cor])
        self.number_of_front_waypoints = len(self.waypoint_front)
        MAP_front = eng.data_processing(True)          # True for Front Waypoints, False for rear waypoints
        #print("This is property of MAP Front",dir(MAP_front.items))
        self.s_front = eng.getfield(MAP_front, "s_int")
        self.xy_front = eng.getfield(MAP_front, "xy")
        self.Heading_front = eng.getfield(MAP_front, "pol_Th")
        self.Curvature_front = eng.getfield(MAP_front, "pol_dTh")

    def PointCollection_Rear(self):
        with open('Rear_Waypoints.txt', 'w') as f:
            self.points_rear =np.zeros((len(self.waypoint_rear),2))
            count_rear=0
            for w in self.waypoint_rear:
                point = w.transform.location
                #print("This is point in the transform front: ", point)
                x_cor = point.x
                y_cor = point.y
                self.points_rear[count_rear][:] = [x_cor, y_cor]
                f.write("%f %f\n" %(x_cor, y_cor))
                #w_front.append([x_cor,y_cor])
        MAP_rear = eng.data_processing(False)

def draw_image(surface, image, blend=False):
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    image_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
    if blend:
        image_surface.set_alpha(100)
    surface.blit(image_surface, (0, 0))

def get_font():
    fonts = [x for x in pygame.font.get_fonts()]
    default_font = 'ubuntumono'
    font = default_font if default_font in fonts else fonts[0]
    font = pygame.font.match_font(font)
    return pygame.font.Font(font, 14)

def should_quit():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return True
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_ESCAPE:
                return True
    return False

def main():
    pygame.init()
    display = pygame.display.set_mode(
        (800, 600),
        pygame.HWSURFACE | pygame.DOUBLEBUF)
    font = get_font()
    clock = pygame.time.Clock()
    #clock.tick_busy_loop(60)
    #clock = pygame.time.Clock()

    #Setting the fps of the pygame simulation
    #clock.tick(20)

    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)

    world = client.get_world()
    world = client.load_world('Town04')

    try:
        actor_list1=[]
        map = world.get_map()
        start_pose = map.get_spawn_points()[1]
        waypoint = map.get_waypoint(start_pose.location)

        blueprint_library = world.get_blueprint_library()

        vehicle = world.spawn_actor(
            blueprint_library.filter("model3")[0],
            start_pose)
        actor_list1.append(vehicle)
        vehicle.set_simulate_physics(False)

        LaneDetection = ImageProcessing()
        Simulation = MySimulation(world,vehicle)
        Simulation.add_sensors(LaneDetection)

        Cars = AddingVehicles(Simulation,start_pose)
        Cars.Vehicle_One()
        Cars.Vehicle_Two()

        camera_spectator = world.spawn_actor(
            blueprint_library.find('sensor.camera.rgb'),
            carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15)),
            attach_to=vehicle)
        actor_list1.append(camera_spectator)


        with CarlaSyncMode(world,camera_spectator,Simulation.camera,Simulation.camera_rear, fps=30) as sync_mode:
            count_for_road_front = 0
            while True:
                if should_quit():
                    return
                clock.tick()
                #print("This is right lane marking :",dir(waypoint.right_lane_marking))

                Road = RoadProcessing(waypoint)
                Road.PointCollection_Front()
                np_xy = np.array(Road.xy_front._data).reshape(Road.xy_front.size, order='F').T
                #print(np_xy)
                #if np_xy.shape[0]>0:
                #    plt.plot(np_xy[1],np_xy[0])
                #    plt.show(block=False)
                #    plt.show()


                waypoint = random.choice(waypoint.next(0.5))
                vehicle.set_transform(waypoint.transform)
                waypoint_Vehicle1 = waypoint
                waypoint_Vehicle2 = waypoint

                ego_velocity = vehicle.get_velocity()
                #Cars.Vehicle1.set_transform(carla.Transform(waypoint.transform.location+carla.Location(x=10*math.cos(vehicle.get_transform().rotation.yaw))))
                #waypoint = random.choice(waypoint.next(6))

                Cars.Vehicle1.set_transform(random.choice(waypoint_Vehicle1.next(10)).transform)
                Cars.Vehicle2.set_transform(random.choice(waypoint_Vehicle2.previous(6)).transform)
                #Cars.Vehicle1.set_velocity(ego_velocity)
                Simulation.world_snapshot = Simulation.world.get_snapshot()

                ego_vehicle_snapshot = Simulation.world_snapshot.find(Simulation.ego_vehicle.id)
                ego_vehicle_state = ego_vehicle_snapshot.get_transform().location

                # Advance the simulation and wait for the data.
                snapshot, image_spectator, image_front, image_rear = sync_mode.tick(timeout=2.0)

                #fps = round(1.0 / snapshot.timestamp.delta_seconds)
                # Draw the display.
                draw_image(display, image_spectator)
                image_for_lane_detection = LaneDetection.make_image_ready(image_front)
                lane_detected_image = LaneDetection.process_image(image_for_lane_detection)
                cv2.imshow("Lane Detected Image Front", lane_detected_image)

                image_for_lane_detection_rear = LaneDetection.make_image_ready(image_rear)
                lane_detected_image_rear = LaneDetection.process_image(image_for_lane_detection_rear)
                cv2.imshow("Lane Detected Image Rear", lane_detected_image_rear)


                #pcd = o3d.io.read_point_cloud('tutorial/new_pcd/%.6d.ply' % Simulation.world_snapshot.timestamp.frame)
                #point_cloud = point_cloud_ply.points
                #print(dir(point_cloud))
                #o3d.visualization.draw_geometries([pcd])

                display.blit(
                    font.render('% 5d FPS (real)' % clock.get_fps(), True, (255, 255, 255)),
                    (8, 10))
                display.blit(
                    font.render('% 5d m (State Of Ego-Vehicle x)' % ego_vehicle_state.x, True, (255, 255, 255)),
                    (8, 46))
                display.blit(
                    font.render('% 5d m (State Of Ego-Vehicle y)' % ego_vehicle_state.y, True, (255, 255, 255)),
                    (8, 64))
                display.blit(
                    font.render('% 5d m (State Of Ego-Vehicle z)' % ego_vehicle_state.z, True, (255, 255, 255)),
                    (8, 82))
                pygame.display.flip()
    finally:
        print('destroying actors.')
        for actor in Simulation.actor_list:
            actor.destroy()
        for actor in actor_list1:
            actor.destroy()
        pygame.quit()
        print('done.')
if __name__ == '__main__':

    try:
        main()

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
