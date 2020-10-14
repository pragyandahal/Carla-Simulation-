
function MAP = data_processing(Front)
    middle2 = PreProcessing(Front);
    mid.x = middle2(:,1);
    mid.y = middle2(:,2);
    mid.s = middle2(:,3);
    hor = 10; % [m] orizzonte spaziale dell'ottimizzazione
    len0 = length(mid.x)-(hor*2+1);

    %% Determinazione tabella mappa Lainate x MPC

    MAP.s_int = zeros(len0,1); % Valore integrale iniziale della s
    MAP.xy = zeros(len0,2);
    MAP.pol_Th = zeros(len0,4);
    MAP.pol_dTh = zeros(len0,4);

    for ii = 1:len0
        s_val = mid.s(ii); % valore attuale s
    %     ii
    %         if ii==2883
    %             keyboard
    %         end
        % inizia la function
        [~,pos] = min(abs(mid.s-s_val)); % punto giusto, dato che il vettore � monotono
        [~,posN] = min(abs(mid.s-(s_val+hor)));

        %% Considero la curva che deve compiere il veicolo

        ascissa = mid.s(pos:posN)-mid.s(pos); % sistema locale
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Costruisco la Ylim(s): prendo per buona la distanza in 5 punti
        % equidistanti e interpolo
        %point_m = [1 2 3 4 6 10 15 22 29 38 42 length(ascissa)]+pos;
        point_m = [1 2 3 4 6 10 length(ascissa)]+pos;
        query_s = ascissa(point_m-point_m(1)+1);
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Costruisco Xs Xss Ys Yss
    %     if point_m(end)>2944
    %         point_m(end)=2944;
    %     end

        [p_xm,S_xm,mu_xm] = polyfit(query_s,mid.x(point_m),3);
        [p_ym,S_ym,mu_ym] = polyfit(query_s,mid.y(point_m),3);
        [p_xsm] = polyder(p_xm);
        [p_ysm] = polyder(p_ym);
        [p_xssm] = polyder(p_xsm);
        [p_yssm] = polyder(p_ysm);

        X_m = polyval(p_xm,ascissa,S_xm,mu_xm);
        Y_m = polyval(p_ym,ascissa,S_ym,mu_ym);
        X_sm = polyval(p_xsm,ascissa,S_xm,mu_xm);
        Y_sm = polyval(p_ysm,ascissa,S_ym,mu_ym);
        X_ssm = polyval(p_xssm,ascissa,S_xm,mu_xm);
        Y_ssm = polyval(p_yssm,ascissa,S_ym,mu_ym);

        Th_c = atan2(Y_sm,X_sm);
        diffe = diff(Th_c);
        dd = find(abs(diffe)>pi);
        if numel(dd)==1
            Th_c(dd+1:end) = Th_c(dd+1:end)-2*pi*sign(diffe(dd));
        elseif numel(dd)>1
            Th_c(dd(1)+1:dd(2)) = Th_c(dd(1)+1:dd(2))-2*pi*sign(diffe(dd(1)));
        end
        pol_Th = polyfit(query_s,Th_c(point_m-point_m(1)+1),3);
    %     Th_cint = polyval(pol_Th,ascissa);
        dTh_c = (X_sm.*Y_ssm - Y_sm.*X_ssm)./((X_sm.^2 + Y_sm.^2).^(3/2));
        pol_dTh = polyfit(query_s,dTh_c(point_m-point_m(1)+1),3);

        %% Creo la mappa passo passo
        MAP.s_int(ii) = s_val; % Valore integrale iniziale della s
        MAP.xy(ii,:) = [mid.x(ii),mid.y(ii)];
        MAP.pol_Th(ii,:) = pol_Th;
        MAP.pol_dTh(ii,:) = pol_dTh;
    end
    %if size(MAP.xy(:,1),1)>40
        %disp("About to plot figure")
        %figure()
        %plot(mid.x,mid.y); hold on
        %scatter(MAP.xy(:,1),MAP.xy(:,2),'.')
        %axis equal
    %end
end


%% Load & Plot
function [middle2]= PreProcessing(Front)
    if Front
        middle1 = load('Front_Waypoints.txt');
    else
        middle1 = load('Front_Waypoints.txt');
    end
    s_mid = zeros(length(middle1),1);
    for ii = 1:length(s_mid)-1
        s_mid(ii+1) = s_mid(ii) + sqrt((middle1(ii+1,2)-middle1(ii,2))^2+...
            (middle1(ii+1,1)-middle1(ii,1))^2);
    end
    middle1(:,3) = s_mid;

    s_new = s_mid(1):0.5:s_mid(end);

    x_new = interp1(s_mid,middle1(:,1),s_new,'linear','extrap');
    y_new = interp1(s_mid,middle1(:,2),s_new,'linear','extrap');

    middle2 = [x_new' y_new' s_new'];
end
