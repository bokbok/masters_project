function display_david(run,tspan, nx, dt)

% \Gamma_ee \Gamma_ei \Gamma_ie \Gamma_ii h_e h_i phi_ee phi_ei
%clear all

to = tspan(1);                 % to in seconds
time = tspan(2);               % total time in seconds
nskip = 25;
cmin = -90;
cmax = -30;
ny = nx;

%fn = 'run.dat.snipped'; % in total 30 s at 0.001 s time resolution
%run = '1375626175';
fn = ['/terra/runs/' run '/run.dat'];
fid = fopen(fn);

textscan(fid,'%*s %*s %*s %*s %*s %*s %*s %*s',1);

%HE = [];
%j=0;
writerObj = VideoWriter(['/terra/runs/' run '_' num2str(to) '_' num2str(time+to) '.avi']);
writerObj.FrameRate = 20;
writerObj.Quality = 100;
open(writerObj);

g1 = axes('position',[0.6667 0.1 0.2333 0.2333]);
g2 = axes('position',[0.6667 0.3833 0.2333 0.2333]);
g3 = axes('position',[0.6667 0.6666 0.2333 0.2333]);
xlabel(g1,'time (s)')
ylabel(g3,'h_e (mV)')

a1 = axes('position',[0.1 0.2167 0.4667 0.5667]);
set(a1,'NextPlot','replaceChildren');%'Visible','off')

for i = 0:(to/dt)-1                    % position file
    T = textscan(fid,'t=%f');
    L = textscan(fid,'( %d,%d ): %f %f %f %f %f %f %f %f',nx*ny);
end

for i = 0:(time/dt)
    T = textscan(fid,'t=%f');
    L = textscan(fid,'( %d,%d ): %f %f %f %f %f %f %f %f',nx*ny);
    if mod(i,nskip)
        continue
    end
    %j=j+1;
    he = L{5};
    he = reshape(he,nx,ny);
    surf(a1,he),shading interp
    zlabel('h_e (mV)','FontSize',14),xlabel('X (grid points)','FontSize',14), ylabel('Y (grid points)','FontSize',14)
    set(gca,'FontSize',14);
    text(-5,10,-60,[num2str(T{1},'%05.3f') ' s'],'FontSize',16,...
        'FontWeight','bold','Color','k')
    axis([1 nx 1 ny -70 -35 -70 -35]); 
    he1 = he(1,1);
    he2 = he(20, 20);
    he3 = he(nx - 1, nx - 1);
    if i
        plot(g1,[(i-nskip)*dt i*dt],[he1l he1],'k-');
        plot(g2,[(i-nskip)*dt i*dt],[he2l he2],'k-');
        plot(g3,[(i-nskip)*dt i*dt],[he3l he3],'k-');
    else
        xlim(g1,[0 time]),ylim(g1,[-70 -35]),hold(g1,'on')
        text(0.05*time,-85,'(1,1)','Parent',g1,'HorizontalAlignment','left');
        xlim(g2,[0 time]),ylim(g2,[-70 -35]),hold(g2,'on')
        text(0.05*time,-85,'(20,20)','Parent',g2,'HorizontalAlignment','left');
        xlim(g3,[0 time]),ylim(g3,[-70 -35]),hold(g3,'on')
        text(0.05*time,-85,'(39,39)','Parent',g3,'HorizontalAlignment','left');
    end
    he1l = he1;
    he2l = he2;
    he3l = he3;
    %scrnsz = get(0, 'ScreenSize');
    %scrnsz = [0 0 420 420];
    %frame = getframe(gcf, scrnsz);
    frame = getframe(gcf);
    writeVideo(writerObj,frame);
    %HE = cat(3,HE,he); % to select a time series assign e.g. dat(1,:) = HE(1,2,:)
    %t(j)=T{1};
end

close(gcf);
fclose(fid);
close(writerObj);