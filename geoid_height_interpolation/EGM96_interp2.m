egmdatnum_1 = importdata('EGM96PolandGeoid.gdf',' ',36);
egmdat_1 = egmdatnum_1.textdata;
egmnum_1 = egmdatnum_1.data;
la_1=egmnum_1(:,1);
fi_1=egmnum_1(:,2);
dat_1=egmnum_1(:,3);
lamin_1=min(la_1);
lamax_1=max(la_1);
fimin_1=min(fi_1);
fimax_1=max(fi_1);


la1=la_1(1);la2=la_1(2);
  i=0;
  while la1==la2 
    i=i+1; 
    la2=la(2+i);
  end 
  fi1=fi_1(1);fi2=fi_1(2);
  j=0;
  while fi1==fi2
    j=j+1;
    fi2=fi_1(2+j);
  end
  dla_1=abs(la1-la2);
  dfi_1=abs(fi1-fi2);
  frows=abs(floor((fimax_1-fimin_1)/dfi_1+1.5));
  lcols=abs(floor((lamax_1-lamin_1)/dla_1+1.5));
  n=1;
  for i=frows:-1:1
    for j=1:lcols
      if n <= length(la_1) 
        if la_1(n) < 180
         X_1(i,j)=la_1(n);
        else
          X_1(i,j)=la_1(n)-360;
        end
      Y_1(i,j)=fi_1(n);
      Z_1(i,j) = dat_1(n);
      n=n+1;
      end
    end    
  end
  
X=X_1(:,1:136);
Y=Y_1(:,1:136);
Z=Z_1(:,1:136);
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