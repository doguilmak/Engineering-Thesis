egmdatnum_2 = importdata('EGM2008PolandGeoid.gdf',' ',36);
egmdat_2 = egmdatnum_2.textdata;
egmnum_2 = egmdatnum_2.data;
la_2=egmnum_2(:,1);
fi_2=egmnum_2(:,2);
dat_2=egmnum_2(:,3);
lamin_2=min(la_2);
lamax_2=max(la_2);
fimin_2=min(fi_2);
fimax_2=max(fi_2);

##plot(la,fi,dat1,'Color',[.4 .4 .4],'LineWidth',1);
##hold on

la1=la_2(1);la2=la_2(2);
  i=0;
  while la1==la2 
    i=i+1; 
    la2=la(2+i);
  end 
  fi1=fi_2(1);fi2=fi_2(2);
  j=0;
  while fi1==fi2
    j=j+1;
    fi2=fi_2(2+j);
  end
  dla_2=abs(la1-la2);
  dfi_2=abs(fi1-fi2);
  frows=abs(floor((fimax_2-fimin_2)/dfi_2+1.5));
  lcols=abs(floor((lamax_2-lamin_2)/dla_2+1.5));
  n=1;
  for i=frows:-1:1
    for j=1:lcols
      if n <= length(la_2) 
        if la_2(n) < 180
         X_2(i,j)=la_2(n);
        else
          X_2(i,j)=la_2(n)-360;
        end
      Y_2(i,j)=fi_2(n);
      Z_2(i,j) = dat_2(n);
      n=n+1;
      end
    end    
  end

X=X_2(:,1:136);
Y=Y_2(:,1:136);
Z=Z_2(:,1:136);
DATA = load('polrefcln.txt');
x=DATA(:,3);
y=DATA(:,2);
z = interp2(X,Y,Z,x,y,'cubic');
standard_deviation = std(z)
surf(X,Y,Z);
hold on
plot3(x,y,z,'ro','MarkerFaceColor','y')
ylabel('Latitude')
xlabel('Longitude')
zlabel('Geoid height(m)')