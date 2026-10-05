#!/usr/bin/env python3
"""CUDA incompressible fluid and inertial ASCII tracers for the Xflops film.

This advances a velocity field, not precomputed paths or target-position lerps:
RK2 semi-Lagrangian velocity transport, viscous diffusion, vorticity confinement,
moving-solid Brinkman coupling, a Fourier pressure Poisson projection, followed
by inertial particle drag, contact impulses and angular response to fluid curl.

The fluid field is periodic in a padded domain; glyph bodies use open outflow.
Input NPZ: xy[N,2], glyph[N], release[N], group[N], size[N]. Obstacle NPZ:
times[T], ellipse[T,6]=(cx,cy,rx,ry,angleRadians,active), anchor_scroll[T].
Output: times[F], xy[F,N,2], velocity[F,N,2], angle[F,N], curl[F,N],
attached[F,N], group[N], size[N], glyph[N], seed_xy[N,2], plus diagnostics.
All spatial values are 1920x1080 screen pixels (Y down); time is seconds.

Run in an isolated CUDA Python environment with cupy-cuda12x, CUDA runtime,
NVRTC and cuFFT component wheels. No graphics context or Blender is needed.
"""
from pathlib import Path
import argparse,json,time,math
import numpy as np

CUDA = r'''
extern "C" {
__device__ float sample(const float* a,float x,float y,int w,int h) {
    x=x-floorf(x/w)*w;y=y-floorf(y/h)*h;
    int ix=(int)floorf(x),iy=(int)floorf(y);float fx=x-ix,fy=y-iy;
    int jx=(ix+1)%w,jy=(iy+1)%h;
    return (1-fy)*((1-fx)*a[iy*w+ix]+fx*a[iy*w+jx])+
           fy*((1-fx)*a[jy*w+ix]+fx*a[jy*w+jx]);
}
__global__ void advect(const float* u,const float* v,float* un,float* vn,
                       int w,int h,float dx,float dt) {
    int id=blockDim.x*blockIdx.x+threadIdx.x;if(id>=w*h)return;
    float x=id%w,y=id/w;
    float mx=x-.5f*dt*u[id]/dx,my=y-.5f*dt*v[id]/dx;
    float um=sample(u,mx,my,w,h),vm=sample(v,mx,my,w,h);
    float bx=x-dt*um/dx,by=y-dt*vm/dx;
    un[id]=sample(u,bx,by,w,h);vn[id]=sample(v,bx,by,w,h);
}
__global__ void anchor_terminal(float* px,float* py,const float* seedx,
                               const float* seedy,const float* release,
                               const int* group,int n,float time,float scroll) {
    int i=blockDim.x*blockIdx.x+threadIdx.x;
    if(i<n && group[i]==0 && time<release[i]) {
        px[i]=seedx[i];py[i]=seedy[i]-scroll;
    }
}
__global__ void release_terminal(const float* px,const float* py,float* release,
                                float* dose,const int* group,const float* radius,
                                const float* ax,const float* ay,
                                const float* u,const float* v,const float* sdf,
                                int n,int w,int h,int mw,int mh,int usemask,
                                float dx,float mx,float my,float dt,float time) {
    int i=blockDim.x*blockIdx.x+threadIdx.x;
    if(i>=n || group[i]!=0 || time>=release[i])return;
    float gx=px[i]/dx-.5f,gy=py[i]/dx-.5f;
    float fu=sample(u,gx,gy,w,h),fv=sample(v,gx,gy,w,h);
    float speed=sqrtf(fu*fu+fv*fv),force=sqrtf(ax[i]*ax[i]+ay[i]*ay[i]);
    // The printed row behaves like lightly adhered paper. An actual solid
    // collision, a neighbour impact, or accumulated fluid drag breaks its
    // attachment. Nothing is released by a global editorial timestamp.
    dose[i]=dose[i]*expf(-dt/.28f)+dt*(fmaxf(speed-95.f,0.f)+.008f*force);
    bool hit=false;
    float sx=(px[i]-mx)/dx-.5f,sy=(py[i]-my)/dx-.5f;
    if(usemask && sx>1 && sx<mw-2 && sy>1 && sy<mh-2)
        hit=sample(sdf,sx,sy,mw,mh)<radius[i]+3.f;
    if(hit || force>2400.f || dose[i]>20.f)release[i]=time;
}
__global__ void tracers(float* px,float* py,float* vx,float* vy,float* angle,
                       float* spin,float* sampled_curl,const float* seedx,
                       const float* seedy,const float* release,const float* tau,const int* group,
                       const float* repelx,const float* repely,const int* gather,
                       const float* u,const float* v,const float* curl,
                       const float* obs,const float* sdf,const float* normalx,
                       const float* normaly,const float* solidu,const float* solidv,
                       int maskw,int maskh,int usemask,float marginx,float marginy,
                       int n,int w,int h,float dx,float dt,
                       float time,float anchor_scroll,float domainw,float domainh) {
    int i=blockDim.x*blockIdx.x+threadIdx.x;if(i>=n)return;
    if(time<release[i]) {
        px[i]=seedx[i];py[i]=seedy[i]-(group[i]==0?anchor_scroll:0);
        vx[i]=0;vy[i]=0;angle[i]=0;spin[i]=0;sampled_curl[i]=0;return;
    }
    float gx=px[i]/dx-.5f,gy=py[i]/dx-.5f;
    float fu=sample(u,gx,gy,w,h),fv=sample(v,gx,gy,w,h);
    vx[i]+=repelx[i]*dt;vy[i]+=repely[i]*dt;
    float drag=1-expf(-dt/tau[i]);
    vx[i]+=(fu-vx[i])*drag;vy[i]+=(fv-vy[i])*drag;
    // A finite-duration external annular potential captures the *existing*
    // screen tracers. This is acceleration plus damping, never a path lerp.
    if(time>=8.05f && time<=9.75f && gather[i]) {
        float ox=px[i]-marginx-960.f,oy=py[i]-marginy-540.f;
        float rr=sqrtf(ox*ox+oy*oy)+.001f,nx=ox/rr,ny=oy/rr;
        float radial=vx[i]*nx+vy[i]*ny,tangent=-vx[i]*ny+vy[i]*nx;
        float ramp=fminf(1.f,fmaxf(0.f,(time-8.05f)/.18f));ramp=ramp*ramp*(3-2*ramp);
        if(gather[i]==1) {
            float target=232.f+33.f*sinf(i*.173f)+10.f*cosf(i*.619f);
            float ar=-440.f*(rr-target)-23.f*radial;
            float target_tangent=510.f+65.f*sinf(atan2f(oy,ox)*2.f);
            float at=40.f*(target_tangent-tangent);
            vx[i]+=dt*ramp*(ar*nx-at*ny);
            vy[i]+=dt*ramp*(ar*ny+at*nx);
        } else if(rr<1800.f) {
            // Non-participating visible IDs leave via an open outflow. They
            // do not vanish through opacity or reappear after a domain wrap.
            float ar=180.f*(1650.f-rr)-26.f*radial;
            vx[i]+=dt*ramp*ar*nx;vy[i]+=dt*ramp*ar*ny;
        }
    }
    px[i]+=vx[i]*dt;py[i]+=vy[i]*dt;
    float om=sample(curl,gx,gy,w,h);
    om=fminf(65.f,fmaxf(-65.f,om));sampled_curl[i]=om;
    spin[i]+=(.5f*om-spin[i])*(1-expf(-dt/.085f));
    angle[i]+=spin[i]*dt*57.2957795f;
    if(usemask) {
        float mx=(px[i]-marginx)/dx-.5f,my=(py[i]-marginy)/dx-.5f;
        if(mx>1 && mx<maskw-2 && my>1 && my<maskh-2) {
            float sd=sample(sdf,mx,my,maskw,maskh);
            if(sd<1.8f) {
                float nx=sample(normalx,mx,my,maskw,maskh),ny=sample(normaly,mx,my,maskw,maskh);
                float norm=sqrtf(nx*nx+ny*ny)+1e-7f;nx/=norm;ny/=norm;
                px[i]+=(1.8f-sd)*nx;py[i]+=(1.8f-sd)*ny;
                // Predictor projection only. The finite-body contact stage
                // applies the moving-wall impulse once per integration step.
            }
        }
    } else if(obs[5]>.01f && obs[2]>1 && obs[3]>1) {
        float c=cosf(obs[4]),s=sinf(obs[4]);
        float ox=px[i]-obs[0],oy=py[i]-obs[1];
        float lx=c*ox+s*oy,ly=-s*ox+c*oy;
        float rx=obs[2]+2.5f,ry=obs[3]+2.5f;
        float q=sqrtf(lx*lx/(rx*rx)+ly*ly/(ry*ry));
        if(q<1.004f) {
            if(q<.00001f){lx=rx;q=1;}
            lx=lx/q*1.005f;ly=ly/q*1.005f;
            px[i]=obs[0]+c*lx-s*ly;py[i]=obs[1]+s*lx+c*ly;
            float nx0=lx/(rx*rx),ny0=ly/(ry*ry);
            float mag=sqrtf(nx0*nx0+ny0*ny0)+1e-8f;
            nx0/=mag;ny0/=mag;
            float nx=c*nx0-s*ny0,ny=s*nx0+c*ny0;
            float sx=obs[8]*lx/rx,sy=obs[9]*ly/ry;
            float bu=obs[6]-obs[10]*(py[i]-obs[1])+c*sx-s*sy;
            float bv=obs[7]+obs[10]*(px[i]-obs[0])+s*sx+c*sy;
            float relx=vx[i]-bu,rely=vy[i]-bv;
            float vn=relx*nx+rely*ny;
            if(vn<0){vx[i]-=1.06f*vn*nx;vy[i]-=1.06f*vn*ny;}
            // Tangential surface friction converts impact into a sliding wake.
            float tangent=relx*(-ny)+rely*nx;
            vx[i]+=.09f*tangent*ny;vy[i]-=.09f*tangent*nx;
        }
    }
    // Open particle outflow: preserve outgoing positions and identity.
    // The velocity grid remains padded and periodic only outside the aperture.
}
__global__ void boundary_seed(const unsigned char* mask,float* sx,float* sy,int w,int h) {
    int id=blockDim.x*blockIdx.x+threadIdx.x;if(id>=w*h)return;
    int x=id%w,y=id/w;unsigned char m=mask[id];
    bool edge=(x>0 && mask[id-1]!=m)||(x<w-1 && mask[id+1]!=m)||
              (y>0 && mask[id-w]!=m)||(y<h-1 && mask[id+w]!=m);
    sx[id]=edge?x:-10000;sy[id]=edge?y:-10000;
}
__global__ void particle_bins(const float* px,const float* py,const float* release,const float* birth,const int* group,
                             int* count,int* ids,int n,int bw,int bh,int capacity,
                             float cell,float time) {
    int i=blockDim.x*blockIdx.x+threadIdx.x;if(i>=n||time<birth[i])return;
    int x=(int)floorf(px[i]/cell),y=(int)floorf(py[i]/cell);
    if(x<0||x>=bw||y<0||y>=bh)return;int b=y*bw+x;
    int slot=atomicAdd(count+b,1);if(slot<capacity)ids[b*capacity+slot]=i;
}
__global__ void contact_forces(const float* px,const float* py,const float* vx,const float* vy,
                              const float* radius,const float* release,const int* group,
                              const int* counts,const int* ids,float* ax,float* ay,
                              int n,int bw,int bh,int capacity,float cell,float time,
                              float domainw,float domainh) {
    int i=blockDim.x*blockIdx.x+threadIdx.x;if(i>=n)return;
    ax[i]=0;ay[i]=0;if(time<release[i] && group[i]!=0)return;
    int bx=((int)floorf(px[i]/cell)%bw+bw)%bw,by=((int)floorf(py[i]/cell)%bh+bh)%bh;
    float fx=0,fy=0;
    for(int oy=-1;oy<=1;oy++)for(int ox=-1;ox<=1;ox++) {
        int x=(bx+ox+bw)%bw,y=(by+oy+bh)%bh,b=y*bw+x;
        int ct=min(counts[b],capacity);
        for(int k=0;k<ct;k++) {
            int j=ids[b*capacity+k];if(j==i)continue;
            float dx=px[i]-px[j],dy=py[i]-py[j];
            // Particle domain is open: outgoing IDs never wrap back into view.
            float distance=sqrtf(dx*dx+dy*dy),contact=radius[i]+radius[j];
            if(distance<contact) {
                if(distance<.001f){float a=(min(i,j)*7+max(i,j)*13)*.618f,sgn=i<j?1.f:-1.f;dx=sgn*cosf(a)*.001f;dy=sgn*sinf(a)*.001f;distance=.001f;}
                float nx=dx/distance,ny=dy/distance;
                float approaching=(vx[i]-vx[j])*nx+(vy[i]-vy[j])*ny;
                float force=600.f*(contact-distance)-14.f*fminf(0.f,approaching);
                fx+=force*nx;fy+=force*ny;
            }
        }
    }
    float mag=sqrtf(fx*fx+fy*fy);float limit=fminf(1.f,24000.f/(mag+1e-6f));
    ax[i]=fx*limit;ay[i]=fy*limit;
}
// Ink rectangles use the same 4-degree rotation quantization as the rasterizer.
__global__ void prepare_shapes(const float* px,const float* py,const float* angle,
                               const float* shape,float* cx,float* cy,float* cs,float* sn,int n) {
 int i=blockDim.x*blockIdx.x+threadIdx.x;if(i>=n)return;
 float a=nearbyintf(angle[i]/4.f)*4.f*.01745329252f,c=cosf(a),s=sinf(a);
 cs[i]=c;sn[i]=s;cx[i]=px[i]+c*shape[i*4+2]+s*shape[i*4+3];
 cy[i]=py[i]-s*shape[i*4+2]+c*shape[i*4+3];
}
__device__ bool ink_contact(int i,int j,const float* cx,const float* cy,
                            const float* cs,const float* sn,const float* shape,
                            float skin,float &pen,float &nx,float &ny) {
 float dx=cx[i]-cx[j],dy=cy[i]-cy[j];
 float hi=shape[i*4]+skin,vi=shape[i*4+1]+skin,hj=shape[j*4]+skin,vj=shape[j*4+1]+skin;
 float ri=sqrtf(hi*hi+vi*vi),rj=sqrtf(hj*hj+vj*vj);
 if(dx*dx+dy*dy>(ri+rj)*(ri+rj))return false;
 float ci=cs[i],si=sn[i],cj=cs[j],sj=sn[j];pen=1e9f;
 float axesx[4]={ci,si,cj,sj},axesy[4]={-si,ci,-sj,cj};
 for(int k=0;k<4;k++) {
  float ax=axesx[k],ay=axesy[k];
  float r1=hi*fabsf(ci*ax-si*ay)+vi*fabsf(si*ax+ci*ay);
  float r2=hj*fabsf(cj*ax-sj*ay)+vj*fabsf(sj*ax+cj*ay);
  float sep=dx*ax+dy*ay,p=r1+r2-fabsf(sep);
  if(p<=0)return false;
  if(p<pen){pen=p;float sign=sep>=0?1.f:-1.f;if(fabsf(sep)<1e-7f)sign=i<j?1.f:-1.f;nx=ax*sign;ny=ay*sign;}
 }
 return true;
}
__device__ bool ink_contact_at(int i,int j,float testx,float testy,const float* cx,const float* cy,
                            const float* cs,const float* sn,const float* shape,
                            float skin,float &pen,float &nx,float &ny) {
 float dx=testx-cx[j],dy=testy-cy[j];
 float hi=shape[i*4]+skin,vi=shape[i*4+1]+skin,hj=shape[j*4]+skin,vj=shape[j*4+1]+skin;
 float ri=sqrtf(hi*hi+vi*vi),rj=sqrtf(hj*hj+vj*vj);
 if(dx*dx+dy*dy>(ri+rj)*(ri+rj))return false;
 float ci=cs[i],si=sn[i],cj=cs[j],sj=sn[j];pen=1e9f;
 float axesx[4]={ci,si,cj,sj},axesy[4]={-si,ci,-sj,cj};
 for(int k=0;k<4;k++) {
  float ax=axesx[k],ay=axesy[k];
  float r1=hi*fabsf(ci*ax-si*ay)+vi*fabsf(si*ax+ci*ay);
  float r2=hj*fabsf(cj*ax-sj*ay)+vj*fabsf(sj*ax+cj*ay);
  float sep=dx*ax+dy*ay,p=r1+r2-fabsf(sep);
  if(p<=0)return false;
  if(p<pen){pen=p;float sign=sep>=0?1.f:-1.f;if(fabsf(sep)<1e-7f)sign=i<j?1.f:-1.f;nx=ax*sign;ny=ay*sign;}
 }
 return true;
}
__device__ unsigned int raster_row_segment(const unsigned int* row,int start) {
 int word=(int)floorf(start/32.f),shift=start-word*32;
 unsigned int a=word>=0&&word<4?row[word]:0,b=word+1>=0&&word+1<4?row[word+1]:0;
 return (a>>shift)|(shift?(b<<(32-shift)):0);
}
__device__ bool raster_contact(int i,int j,float x,float y,const float* px,const float* py,
 const float* cs,const float* sn,const int* raster_ids,const int* offsets,const int* widths,const int* heights,const unsigned int* bits,float marginx,float marginy) {
 int ai=(int)nearbyintf(atan2f(sn[i],cs[i])*14.323944878f);ai=(ai%90+90)%90;
 int aj=(int)nearbyintf(atan2f(sn[j],cs[j])*14.323944878f);aj=(aj%90+90)%90;
 int ti=raster_ids[i]*90+ai,tj=raster_ids[j]*90+aj;
 int wi=widths[ti],hi=heights[ti],wj=widths[tj],hj=heights[tj];
 // Match exported float32 camera coordinates, then Python's double-precision
 // round() at the paste anchor. Translating before float rounding is not
 // invariant at half-pixel boundaries.
 float iwx=__fsub_rn(x,marginx),iwy=__fsub_rn(y,marginy);
 float jwx=__fsub_rn(px[j],marginx),jwy=__fsub_rn(py[j],marginy);
 int ix=(int)nearbyint((double)iwx-wi*.5)+(int)marginx,iy=(int)nearbyint((double)iwy-hi*.5)+(int)marginy;
 int jx=(int)nearbyint((double)jwx-wj*.5)+(int)marginx,jy=(int)nearbyint((double)jwy-hj*.5)+(int)marginy;
 if(ix>=jx+wj||jx>=ix+wi||iy>=jy+hj||jy>=iy+hi)return false;
 int top=max(iy,jy),bottom=min(iy+hi,jy+hj);
 for(int yy=top;yy<bottom;yy++) {
  const unsigned int* ir=bits+offsets[ti]+(yy-iy)*4;
  const unsigned int* jr=bits+offsets[tj]+(yy-jy)*4;
  for(int word=0;word<4;word++)if(ir[word]&raster_row_segment(jr,ix+word*32-jx))return true;
 }
 return false;
}

__device__ bool contact_position_free(int i,float x,float y,const float* px,const float* py,
 const float* cx,const float* cy,const float* cs,const float* sn,const float* shape,const float* birth,
 const int* counts,const int* ids,int bw,int bh,int capacity,float cell,float time,float skin,
 const int* band,const float* sdf,int mw,int mh,float dx,float marginx,float marginy,
 const int* raster_ids,const int* offsets,const int* widths,const int* heights,const unsigned int* bits) {
 float ox=x-px[i],oy=y-py[i],icx=cx[i]+ox,icy=cy[i]+oy;
 float gx=(icx-marginx)/dx-.5f,gy=(icy-marginy)/dx-.5f;
 if(gx>1&&gx<mw-2&&gy>1&&gy<mh-2&&sample(sdf+band[i]*mw*mh,gx,gy,mw,mh)<.75f)return false;
 int bx=(int)floorf(x/cell),by=(int)floorf(y/cell);
 // Membership is updated immediately after every serial recovery move.
 // A second ring also covers center offsets and raster expansion.
 for(int oy=-2;oy<=2;oy++)for(int ox=-2;ox<=2;ox++) {
  int xx=bx+ox,yy=by+oy;if(xx<0||xx>=bw||yy<0||yy>=bh)continue;
  int b=yy*bw+xx,ct=min(counts[b],capacity);
  for(int k=0;k<ct;k++) {
   int j=ids[b*capacity+k];if(j==i||time<birth[j])continue;
   float pen,nx,ny;if(ink_contact_at(i,j,icx,icy,cx,cy,cs,sn,shape,skin,pen,nx,ny)&&raster_contact(i,j,x,y,px,py,cs,sn,raster_ids,offsets,widths,heights,bits,marginx,marginy))return false;
  }
 }
 return true;
}
__global__ void flag_recovery(const float* px,const float* py,const float* cx,const float* cy,
 const float* cs,const float* sn,const float* shape,const float* release,const float* birth,
 const int* counts,const int* ids,int n,int bw,int bh,int capacity,float cell,float time,float skin,
 const int* band,const float* sdf,int mw,int mh,float dx,float marginx,float marginy,
 const int* raster_ids,const int* offsets,const int* widths,const int* heights,const unsigned int* bits,int* needs) {
 int i=blockDim.x*blockIdx.x+threadIdx.x;if(i>=n)return;needs[i]=0;
 if(time<birth[i]||time<release[i])return;
 float x=px[i],y=py[i];if(x<0||y<0||x>=bw*cell||y>=bh*cell)return;
 needs[i]=!contact_position_free(i,x,y,px,py,cx,cy,cs,sn,shape,birth,counts,ids,bw,bh,capacity,cell,time,skin,band,sdf,mw,mh,dx,marginx,marginy,raster_ids,offsets,widths,heights,bits);
}

__global__ void recover_contacts(float* px,float* py,float* vx,float* vy,float* spin,
 float* cx,float* cy,const float* cs,const float* sn,const float* shape,const float* release,const float* birth,
 int* counts,int* ids,int n,int bw,int bh,int capacity,float cell,float time,float skin,
 const int* band,const float* sdf,int mw,int mh,float dx,float marginx,float marginy,float invdt,
 float* distance,int* failed,int reverse,const int* needs,
 const int* raster_ids,const int* offsets,const int* widths,const int* heights,const unsigned int* bits) {
 if(blockIdx.x||blockDim.x!=32)return;int lane=threadIdx.x;
 for(int k=lane;k<n;k+=32){distance[k]=0;failed[k]=0;}
 __syncwarp();
 for(int k=0;k<n;k++) {
  int i=reverse?n-1-k:k;if(!needs[i]||time<birth[i]||time<release[i])continue;
  float x=px[i],y=py[i];if(x<0||y<0||x>=bw*cell||y>=bh*cell)continue;
  int blocked=0;
  if(lane==0)blocked=!contact_position_free(i,x,y,px,py,cx,cy,cs,sn,shape,birth,counts,ids,bw,bh,capacity,cell,time,skin,band,sdf,mw,mh,dx,marginx,marginy,raster_ids,offsets,widths,heights,bits);
  blocked=__shfl_sync(0xffffffff,blocked,0);if(!blocked)continue;
  bool found=false;float movedx=0,movedy=0;
  float radii[19]={.125f,.25f,.5f,1.f,1.5f,2.f,3.f,4.f,6.f,8.f,12.f,16.f,24.f,40.f,64.f,80.f,96.f,128.f,192.f};
  float preferred=atan2f(vy[i],vx[i]);
  for(int ring=0;ring<19&&!found;ring++) {
   // One warp tests the original 32 ordered directions concurrently.
   // The lowest feasible lane is exactly the first-valid serial candidate.
   int zig=(lane+1)/2*(lane%2?1:-1);float a=preferred+zig*.19634954085f;
   float sx=radii[ring]*cosf(a),sy=radii[ring]*sinf(a);
   bool valid=contact_position_free(i,x+sx,y+sy,px,py,cx,cy,cs,sn,shape,birth,counts,ids,bw,bh,capacity,cell,time,skin,band,sdf,mw,mh,dx,marginx,marginy,raster_ids,offsets,widths,heights,bits);
   unsigned int ballot=__ballot_sync(0xffffffff,valid);
   if(ballot) {
    int first=__ffs(ballot)-1;movedx=__shfl_sync(0xffffffff,sx,first);movedy=__shfl_sync(0xffffffff,sy,first);found=true;
   }
  }
  if(found && lane==0) {
   int oldx=(int)floorf(px[i]/cell),oldy=(int)floorf(py[i]/cell);
   px[i]+=movedx;py[i]+=movedy;cx[i]+=movedx;cy[i]+=movedy;
   int newx=(int)floorf(px[i]/cell),newy=(int)floorf(py[i]/cell);
   if(oldx!=newx||oldy!=newy) {
    int oldbin=oldy*bw+oldx,ct=min(counts[oldbin],capacity);
    for(int q=0;q<ct;q++)if(ids[oldbin*capacity+q]==i){ids[oldbin*capacity+q]=ids[oldbin*capacity+ct-1];counts[oldbin]=ct-1;break;}
    if(newx>=0&&newx<bw&&newy>=0&&newy<bh){int b=newy*bw+newx,slot=counts[b];if(slot<capacity){ids[b*capacity+slot]=i;counts[b]++;}}
   }
   vx[i]+=.08f*movedx*invdt;vy[i]+=.08f*movedy*invdt;spin[i]*=.9f;
   distance[i]=sqrtf(movedx*movedx+movedy*movedy);
  } else if(!found&&lane==0)failed[i]=1;
  __syncwarp();
 }
}

__device__ bool feasible_contact(int i,int j,const float* cx,const float* cy,const float* cs,const float* sn,
 const float* shape,float skin,float wi,float wj,const int* band,const float* sdf,const float* gx,const float* gy,
 int mw,int mh,float dx,float marginx,float marginy,float &dix,float &diy,float &djx,float &djy,float &penetration) {
 float nx,ny;if(!ink_contact(i,j,cx,cy,cs,sn,shape,skin,penetration,nx,ny))return false;
 float bnx[2]={0,0},bny[2]={0,0};int who[2]={i,j};
 for(int q=0;q<2;q++) {
  int p=who[q];float x=(cx[p]-marginx)/dx-.5f,y=(cy[p]-marginy)/dx-.5f;
  if(x>1&&x<mw-2&&y>1&&y<mh-2) {
   int offset=band[p]*mw*mh;float clearance=sample(sdf+offset,x,y,mw,mh);
   if(clearance<12.f) {
    float xx=sample(gx+offset,x,y,mw,mh),yy=sample(gy+offset,x,y,mw,mh),len=sqrtf(xx*xx+yy*yy);
    if(len>1e-6f){bnx[q]=xx/len;bny[q]=yy/len;}
   }
  }
 }
 float ci=cs[i],si=sn[i],cj=cs[j],sj=sn[j],hi=shape[4*i]+skin,vi=shape[4*i+1]+skin,hj=shape[4*j]+skin,vj=shape[4*j+1]+skin;
 float axesx[4]={ci,si,cj,sj},axesy[4]={-si,ci,-sj,cj},best=1e30f;
 dix=diy=djx=djy=0;
 for(int a=0;a<4;a++)for(int sign=-1;sign<=1;sign+=2) {
  float x=axesx[a]*sign,y=axesy[a]*sign;
  float r1=hi*fabsf(ci*x-si*y)+vi*fabsf(si*x+ci*y),r2=hj*fabsf(cj*x-sj*y)+vj*fabsf(sj*x+cj*y);
  float pen=r1+r2-((cx[i]-cx[j])*x+(cy[i]-cy[j])*y)+.025f;
  float ix=x,iy=y,jx=-x,jy=-y;
  float inward=fminf(0.f,ix*bnx[0]+iy*bny[0]);ix-=inward*bnx[0];iy-=inward*bny[0];
  inward=fminf(0.f,jx*bnx[1]+jy*bny[1]);jx-=inward*bnx[1];jy-=inward*bny[1];
  float denom=wi*(ix*x+iy*y)+wj*(jx*(-x)+jy*(-y));if(denom<.025f)continue;
  float cost=pen*pen/denom;if(cost>=best)continue;best=cost;
  float lambda=pen/denom;dix=lambda*wi*ix;diy=lambda*wi*iy;djx=lambda*wj*jx;djy=lambda*wj*jy;
 }
 return true;
}

// Nine-color Gauss-Seidel: same-color cells have disjoint 3x3 neighborhoods.
// Each thread resolves one cell serially, updating both participants. Unlike
// a spring or averaged Jacobi push, a contact leaves no penetration when solved.
__global__ void solve_contacts_colored(float* px,float* py,float* vx,float* vy,float* spin,
 float* cx,float* cy,const float* cs,const float* sn,const float* shape,
 float* release,const float* birth,const int* group,const int* counts,const int* ids,
 int n,int bw,int bh,int capacity,float cell,float time,float skin,int color,float invdt,float velocity_gain,
 const int* band,const float* sdf,const float* normalx,const float* normaly,int mw,int mh,float dx,float marginx,float marginy) {
 int b=blockDim.x*blockIdx.x+threadIdx.x;if(b>=bw*bh)return;
 int bx=b%bw,by=b/bw;if((bx%3)+3*(by%3)!=color)return;
 int count=min(counts[b],capacity);
 for(int ki=0;ki<count;ki++) {
  int i=ids[b*capacity+ki];if(time<birth[i])continue;
  float wi=time<release[i]?0.f:1.f;
  for(int oy=-1;oy<=1;oy++)for(int ox=-1;ox<=1;ox++) {
   int x=bx+ox,y=by+oy;if(x<0||x>=bw||y<0||y>=bh)continue;
   int bb=y*bw+x,len=min(counts[bb],capacity);
   for(int kj=0;kj<len;kj++) {
    int j=ids[bb*capacity+kj];if(j<=i||time<birth[j])continue;
    float wj=time<release[j]?0.f:1.f,total=wi+wj;if(total==0)continue;
    float pen,nx,ny;if(!ink_contact(i,j,cx,cy,cs,sn,shape,skin,pen,nx,ny))continue;
    // A real finite glyph impact peels an adhered terminal neighbor. The
    // earlier circle-force estimate alone cannot detect wide/rotated ink.
    float approaching=(vx[i]-vx[j])*nx+(vy[i]-vy[j])*ny;
    if(pen>1.5f || approaching<-40.f) {
      if(wi==0.f&&group[i]==0){release[i]=time;wi=1.f;}
      if(wj==0.f&&group[j]==0){release[j]=time;wj=1.f;}
      total=wi+wj;
    }
    float dix,diy,djx,djy;
    feasible_contact(i,j,cx,cy,cs,sn,shape,skin,wi,wj,band,sdf,normalx,normaly,mw,mh,dx,marginx,marginy,dix,diy,djx,djy,pen);
    px[i]+=dix;py[i]+=diy;cx[i]+=dix;cy[i]+=diy;
    px[j]+=djx;py[j]+=djy;cx[j]+=djx;cy[j]+=djy;
    vx[i]+=dix*invdt*velocity_gain;vy[i]+=diy*invdt*velocity_gain;
    vx[j]+=djx*invdt*velocity_gain;vy[j]+=djy*invdt*velocity_gain;
    if(velocity_gain>0){spin[i]*=.97f;spin[j]*=.97f;}
   }
  }
 }
}

__global__ void project_contacts(const float* px,const float* py,const float* cx,const float* cy,
 const float* cs,const float* sn,const float* shape,const float* release,const float* birth,const int* group,
 const int* counts,const int* ids,float* shiftx,float* shifty,float* penetration,int* hits,
 int n,int bw,int bh,int capacity,float cell,float time,float skin,
 const int* band,const float* sdf,const float* normalx,const float* normaly,int mw,int mh,float dx,float marginx,float marginy) {
 int i=blockDim.x*blockIdx.x+threadIdx.x;if(i>=n)return;
 shiftx[i]=0;shifty[i]=0;penetration[i]=0;hits[i]=0;
 if(time<birth[i])return;
 int bx=(int)floorf(px[i]/cell),by=(int)floorf(py[i]/cell);
 if(bx<0||bx>=bw||by<0||by>=bh)return;
 float sx=0,sy=0,maximum=0;int ct=0;
 bool fixed=time<release[i];
 for(int oy=-1;oy<=1;oy++)for(int ox=-1;ox<=1;ox++) {
  int x=bx+ox,y=by+oy;if(x<0||x>=bw||y<0||y>=bh)continue;
  int b=y*bw+x,len=min(counts[b],capacity);
  for(int k=0;k<len;k++) {
   int j=ids[b*capacity+k];if(j==i||time<birth[j]||(fixed&&time<release[j]))continue;
   float pen,nx,ny;if(!ink_contact(i,j,cx,cy,cs,sn,shape,skin,pen,nx,ny))continue;
   maximum=fmaxf(maximum,pen);ct++;
   if(fixed)continue;
   float dix,diy,djx,djy;
   feasible_contact(i,j,cx,cy,cs,sn,shape,skin,1.f,time<release[j]?0.f:1.f,band,sdf,normalx,normaly,mw,mh,dx,marginx,marginy,dix,diy,djx,djy,pen);
   sx+=dix;sy+=diy;
  }
 }
 // Jacobi averaging is stable even in a squeezed crowd; repeated iterations
 // propagate pressure through the entire pile instead of letting ink cross.
 float relax=1.f/fmaxf(1.f,ct*.55f);
 shiftx[i]=sx*relax;shifty[i]=sy*relax;penetration[i]=maximum;hits[i]=ct;
}
__global__ void apply_contacts(float* px,float* py,float* vx,float* vy,float* spin,
 const float* shiftx,const float* shifty,const int* hits,int n,float invdt,float velocity_gain) {
 int i=blockDim.x*blockIdx.x+threadIdx.x;if(i>=n)return;
 float sx=shiftx[i],sy=shifty[i];px[i]+=sx;py[i]+=sy;
 vx[i]+=sx*invdt*velocity_gain;vy[i]+=sy*invdt*velocity_gain;
 if(hits[i]&&velocity_gain>0)spin[i]*=.96f;
}
__global__ void project_solids(float* px,float* py,float* vx,float* vy,float* spin,
 const float* angle,const float* shape,const float* release,const float* birth,const int* band,
 const float* sdf,const float* normalx,const float* normaly,const float* solidu,const float* solidv,
 int n,int mw,int mh,float dx,float marginx,float marginy,float time,int velocity_response) {
 int i=blockDim.x*blockIdx.x+threadIdx.x;if(i>=n||time<birth[i]||time<release[i])return;
 float a=nearbyintf(angle[i]/4.f)*4.f*.01745329252f,c=cosf(a),s=sinf(a);
 float ox=shape[i*4+2],oy=shape[i*4+3];int offset=band[i]*mw*mh;
 for(int pass=0;pass<3;pass++) {
  float centerx=px[i]+c*ox+s*oy,centery=py[i]-s*ox+c*oy;
  float gx=(centerx-marginx)/dx-.5f,gy=(centery-marginy)/dx-.5f;
  if(gx<1||gx>=mw-2||gy<1||gy>=mh-2)break;
  // This signed distance was REINITIALIZED after dilating solid geometry
  // by the body's conservative ink radius. Its normal points to reachable
  // free centers, not toward a local maximum inside an impossible thin gap.
  float sd=sample(sdf+offset,gx,gy,mw,mh);if(sd>=1.f)break;
  float nx=sample(normalx+offset,gx,gy,mw,mh),ny=sample(normaly+offset,gx,gy,mw,mh);
  float norm=sqrtf(nx*nx+ny*ny);if(norm<1e-6f)break;nx/=norm;ny/=norm;
  px[i]+=(1.f-sd)*nx;py[i]+=(1.f-sd)*ny;
  // Re-solving a positional constraint is not another physical collision.
  // Repeated reflections against changing corner normals inject energy.
  if(velocity_response && pass==0) {
   float bu=sample(solidu,gx,gy,mw,mh),bv=sample(solidv,gx,gy,mw,mh);
   float inward=(vx[i]-bu)*nx+(vy[i]-bv)*ny;
   if(inward<0){vx[i]-=1.02f*inward*nx;vy[i]-=1.02f*inward*ny;}
   float tangent=(vx[i]-bu)*(-ny)+(vy[i]-bv)*nx;
   vx[i]+=.025f*tangent*ny;vy[i]-=.025f*tangent*nx;
   spin[i]*=.94f;
  }
 }
}

__global__ void jump_flood(const float* sx,const float* sy,float* tx,float* ty,int w,int h,int step) {
    int id=blockDim.x*blockIdx.x+threadIdx.x;if(id>=w*h)return;
    int x=id%w,y=id/w;float bx=sx[id],by=sy[id];
    float best=(bx-x)*(bx-x)+(by-y)*(by-y);
    for(int dy=-1;dy<=1;dy++)for(int dx=-1;dx<=1;dx++) {
        int xx=x+dx*step,yy=y+dy*step;if(xx<0||xx>=w||yy<0||yy>=h)continue;
        int j=yy*w+xx;float cx=sx[j],cy=sy[j];float d=(cx-x)*(cx-x)+(cy-y)*(cy-y);
        if(d<best){best=d;bx=cx;by=cy;}
    }
    tx[id]=bx;ty[id]=by;
}
}
'''


def smooth(x):
    x=np.clip(x,0,1);return x*x*(3-2*x)


class Fluid:
    def __init__(self,cp,w=1536,h=432,dx=3.75,hz=120,viscosity=11.):
        self.cp=cp;self.w=w;self.h=h;self.dx=float(dx);self.dt=1/hz
        self.viscosity=viscosity
        shape=(h,w)
        self.u=cp.zeros(shape,cp.float32);self.v=cp.zeros_like(self.u)
        self.un=cp.empty_like(self.u);self.vn=cp.empty_like(self.u)
        self.x,self.y=cp.meshgrid((cp.arange(w,dtype=cp.float32)+.5)*dx,
                                 (cp.arange(h,dtype=cp.float32)+.5)*dx)
        # Projection uses the exact Fourier symbol of the centered discrete
        # divergence/gradient; the post-projection divergence is measurable.
        fx=cp.fft.rfftfreq(w).astype(cp.float32);fy=cp.fft.fftfreq(h).astype(cp.float32)
        self.kx=(cp.sin(2*cp.pi*fx)/dx)[None,:]
        self.ky=(cp.sin(2*cp.pi*fy)/dx)[:,None]
        self.kx[0,-1]=0;self.ky[h//2,0]=0
        self.k2=self.kx*self.kx+self.ky*self.ky
        self.inv_k2=cp.where(self.k2>1e-12,1/cp.maximum(self.k2,1e-12),0).astype(cp.float32)
        real_k2=(2*cp.pi*fx[None,:]/dx)**2+(2*cp.pi*fy[:,None]/dx)**2
        self.diffuse=cp.exp(-viscosity*self.dt*real_k2).astype(cp.float32)
        module=cp.RawModule(code=CUDA,options=('--std=c++11',),name_expressions=('advect','tracers','boundary_seed','jump_flood','particle_bins','contact_forces','anchor_terminal','release_terminal','prepare_shapes','project_contacts','apply_contacts','project_solids','solve_contacts_colored','flag_recovery','recover_contacts'))
        self.advect=module.get_function('advect');self.tracers=module.get_function('tracers')
        self.obs=cp.zeros(11,cp.float32)
        self.curl=cp.zeros_like(self.u)
        self.seed_kernel=module.get_function('boundary_seed');self.jump_kernel=module.get_function('jump_flood')
        self.bin_kernel=module.get_function('particle_bins');self.contact_kernel=module.get_function('contact_forces')
        self.anchor_kernel=module.get_function('anchor_terminal');self.release_kernel=module.get_function('release_terminal')
        self.prepare_shapes=module.get_function('prepare_shapes');self.project_contacts=module.get_function('project_contacts')
        self.apply_contacts=module.get_function('apply_contacts');self.project_solids=module.get_function('project_solids')
        self.solve_contacts_colored=module.get_function('solve_contacts_colored')
        self.flag_recovery=module.get_function('flag_recovery')
        self.recover_contacts=module.get_function('recover_contacts')
        self.maskw=512;self.maskh=288
        self.solidmask=cp.zeros((self.maskh,self.maskw),cp.uint8)
        self.sdf=cp.full((self.maskh,self.maskw),1e4,cp.float32)
        self.normalx=cp.zeros_like(self.sdf);self.normaly=cp.zeros_like(self.sdf)
        self.solidu=cp.zeros_like(self.sdf);self.solidv=cp.zeros_like(self.sdf)
        self.usemask=False
        self.clearance_radii=np.array([8.,12.,16.,20.,24.,28.,34.,40.],np.float32)
        self.clearance_sdf=cp.full((len(self.clearance_radii),self.maskh,self.maskw),1e4,cp.float32)
        self.clearance_normalx=cp.zeros_like(self.clearance_sdf);self.clearance_normaly=cp.zeros_like(self.clearance_sdf)
        self.contact_solidu=cp.zeros_like(self.sdf);self.contact_solidv=cp.zeros_like(self.sdf)
        self.diag=[]

    def update_solid_mask(self,mask,velocity):
        """GPU jump-flood signed distance to the actual articulated silhouette."""
        cp=self.cp;self.usemask=True
        # Keep host buffers alive through asynchronous device transfer. A
        # short-lived float16->float32 temporary must never feed stale memory.
        self._solid_host=(np.ascontiguousarray(mask,dtype=np.uint8),
                          np.ascontiguousarray(velocity[...,0],dtype=np.float32),
                          np.ascontiguousarray(velocity[...,1],dtype=np.float32))
        self.solidmask.set(self._solid_host[0])
        self.solidu.set(self._solid_host[1]);self.solidv.set(self._solid_host[2])
        cp.cuda.get_current_stream().synchronize()
        if not np.any(mask):
            self.sdf.fill(1e4);self.normalx.fill(0);self.normaly.fill(0);self.clearance_sdf.fill(1e4);self.clearance_normalx.fill(0);self.clearance_normaly.fill(0);return
        # The camera aperture is not a physical wall. Build distances with
        # genuine free exterior, then crop back to the unchanged mask origin.
        # Padding exceeds the largest finite-body clearance radius (40px).
        pad=16;ph=self.maskh+2*pad;pw=self.maskw+2*pad
        core=(slice(pad,pad+self.maskh),slice(pad,pad+self.maskw))
        padded=cp.pad(self.solidmask,pad,mode='constant',constant_values=0)
        sx=cp.empty((ph,pw),cp.float32);sy=cp.empty_like(sx)
        tx=cp.empty_like(sx);ty=cp.empty_like(sx)
        blocks=((pw*ph+255)//256,)
        yy,xx=cp.indices((ph,pw),dtype=cp.float32)
        jumps=[2**power for power in range(int(math.ceil(math.log2(max(ph,pw))))-1,-1,-1)]+[1]

        def distance_field(occupied):
            nonlocal sx,sy,tx,ty
            self.seed_kernel(blocks,(256,),(occupied,sx,sy,np.int32(pw),np.int32(ph)))
            for jump in jumps:
                self.jump_kernel(blocks,(256,),(sx,sy,tx,ty,np.int32(pw),np.int32(ph),np.int32(jump)))
                sx,tx=tx,sx;sy,ty=ty,sy
            distance=cp.sqrt((xx-sx)**2+(yy-sy)**2)*self.dx
            return (distance+.5*self.dx)*cp.where(occupied>0,-1.,1.).astype(cp.float32)

        def normals(field):
            # Central interior differences and one-sided exterior differences;
            # no periodic wrap from one camera edge into the opposite edge.
            nx=cp.empty_like(field);ny=cp.empty_like(field)
            nx[:,1:-1]=(field[:,2:]-field[:,:-2])/(2*self.dx)
            nx[:,0]=(field[:,1]-field[:,0])/self.dx
            nx[:,-1]=(field[:,-1]-field[:,-2])/self.dx
            ny[1:-1]=(field[2:]-field[:-2])/(2*self.dx)
            ny[0]=(field[1]-field[0])/self.dx
            ny[-1]=(field[-1]-field[-2])/self.dx
            norm=cp.sqrt(nx*nx+ny*ny+1e-8)
            return nx/norm,ny/norm

        base=distance_field(padded);nx,ny=normals(base)
        self.sdf=cp.ascontiguousarray(base[core])
        self.normalx=cp.ascontiguousarray(nx[core]);self.normaly=cp.ascontiguousarray(ny[core])
        # Extend boundary motion using the same original raster coordinates.
        bx=cp.clip(cp.rint(sx[core]).astype(cp.int32)-pad,0,self.maskw-1)
        by=cp.clip(cp.rint(sy[core]).astype(cp.int32)-pad,0,self.maskh-1)
        ix=cp.clip(bx-cp.rint(2*self.normalx[by,bx]).astype(cp.int32),0,self.maskw-1)
        iy=cp.clip(by-cp.rint(2*self.normaly[by,bx]).astype(cp.int32),0,self.maskh-1)
        self.contact_solidu=self.solidu[iy,ix];self.contact_solidv=self.solidv[iy,ix]
        for band,radius in enumerate(self.clearance_radii):
            expanded=(base<radius+1.0).astype(cp.uint8)
            field=distance_field(expanded);nx,ny=normals(field)
            self.clearance_sdf[band]=field[core]
            self.clearance_normalx[band]=nx[core];self.clearance_normaly[band]=ny[core]

    def curl_field(self):
        cp=self.cp;d=self.dx
        return ((cp.roll(self.v,-1,axis=1)-cp.roll(self.v,1,axis=1))-
                (cp.roll(self.u,-1,axis=0)-cp.roll(self.u,1,axis=0)))/(2*d)

    def projection(self,diagnostic=False):
        cp=self.cp
        uh=cp.fft.rfft2(self.u);vh=cp.fft.rfft2(self.v)
        dot=self.kx*uh+self.ky*vh
        pressure_gradient=dot*self.inv_k2
        uh=(uh-self.kx*pressure_gradient)*self.diffuse
        vh=(vh-self.ky*pressure_gradient)*self.diffuse
        self.u=cp.fft.irfft2(uh,s=(self.h,self.w)).astype(cp.float32)
        self.v=cp.fft.irfft2(vh,s=(self.h,self.w)).astype(cp.float32)
        if diagnostic:
            before=cp.fft.irfft2(1j*dot,s=(self.h,self.w))
            after=cp.fft.irfft2(1j*(self.kx*uh+self.ky*vh),s=(self.h,self.w))
            return float(cp.sqrt(cp.mean(before*before)).get()),float(cp.sqrt(cp.mean(after*after)).get())

    def step(self,t,obstacle,margin,diagnostic=False):
        cp=self.cp;dt=self.dt;dx=self.dx
        blocks=((self.w*self.h+255)//256,)
        self.advect(blocks,(256,),(self.u,self.v,self.un,self.vn,np.int32(self.w),np.int32(self.h),np.float32(dx),np.float32(dt)))
        self.u,self.un=self.un,self.u;self.v,self.vn=self.vn,self.v
        xx=self.x-margin[0];yy=self.y-margin[1]
        # A physical inlet pressure/wind drive supplies continuous material
        # from the off-screen reservoir; no particle is teleported into view.
        target=float(np.interp(t,[3,3.3,4.1,4.45,5.2,6.4,7.60,7.85,8.3,9.7,11.3,13.,15.,16.2,16.7,17.,18.,18.8,19.75],
                                [0,0,0,-540,-720,-710,-670,-790,-190,-180,-380,-550,-540,-180,700,850,900,550,250]))
        drive=2.2 if 16.2<t<18.4 else .63
        self.u+=(target-self.u)*(1-math.exp(-dt*drive))
        self.v*=math.exp(-dt*.04)
        # Opening vorticity now comes only from the visible moving RUN solid.
        # No independent fan impulses are placed underneath untouched rows.
        # The camera follows the accelerator into the cluster. The existing
        # current clears to the left; there is no invisible annular gather or
        # central rotor. Wakes now come from the visible docks and processors.
        track=float(smooth((t-7.30)/.65)*(1-smooth((t-9.35)/.85)))
        self.u+=dt*track*(-1050.-self.u)*1.7
        output_jet=math.exp(-((t-16.67)/.31)**2)
        if output_jet>.002:
            self.v+=dt*output_jet*1150*cp.exp(-((xx-960)/850)**2)
        curl=self.curl_field()
        absolute=cp.abs(curl)
        gx=(cp.roll(absolute,-1,1)-cp.roll(absolute,1,1))/(2*dx)
        gy=(cp.roll(absolute,-1,0)-cp.roll(absolute,1,0))/(2*dx)
        inv=1/cp.sqrt(gx*gx+gy*gy+1e-8)
        self.u+=dt*(2.3*dx)*gy*inv*curl
        self.v-=dt*(2.3*dx)*gx*inv*curl
        # Immersed moving-solid no-slip penalty. Translation, rotation and
        # ellipse growth all supply boundary velocity and displace the fluid.
        o=np.asarray(obstacle,dtype=np.float32).copy();o[:2]+=margin
        self.obs.set(o)
        if self.usemask:
            iy=int(margin[1]/dx);ix=int(margin[0]/dx)
            patch=(slice(iy,iy+self.maskh),slice(ix,ix+self.maskw))
            band=cp.clip((2.7-self.sdf)/7.5,0,1)
            penalty=1-cp.exp(-dt*155*band)
            self.u[patch]+=(self.solidu-self.u[patch])*penalty
            self.v[patch]+=(self.solidv-self.v[patch])*penalty
        elif o[5]>.01 and min(o[2:4])>1:
            c=math.cos(float(o[4]));s=math.sin(float(o[4]))
            ox=self.x-o[0];oy=self.y-o[1]
            lx=c*ox+s*oy;ly=-s*ox+c*oy
            q=cp.sqrt((lx/o[2])**2+(ly/o[3])**2)
            band=cp.clip((1.023-q)/.045,0,1)
            penalty=(1-cp.exp(-dt*155*band))*o[5]
            sx=o[8]*lx/o[2];sy=o[9]*ly/o[3]
            solidu=o[6]-o[10]*oy+c*sx-s*sy
            solidv=o[7]+o[10]*ox+s*sx+c*sy
            self.u+=(solidu-self.u)*penalty
            self.v+=(solidv-self.v)*penalty
        info=self.projection(diagnostic)
        self.curl=self.curl_field()
        if diagnostic:
            speed=cp.sqrt(self.u*self.u+self.v*self.v)
            self.diag.append((t,*info,float(cp.max(speed).get()),float(cp.sqrt(cp.mean(self.curl**2)).get())))
        return info


def obstacle_track(path,times):
    if not path:return np.zeros((len(times),11),np.float32),np.zeros(len(times),np.float32)
    data=np.load(path);source_t=data['times'];ellipse=data['ellipse'].astype(float)
    arr=np.column_stack([np.interp(times,source_t,ellipse[:,j]) for j in range(6)])
    # Discontinuous source cuts have independent hardware bodies. Do not turn
    # a cut's position jump into a fictitious infinite object velocity.
    derivative=np.gradient(arr[:,:5],times,axis=0)
    jumps=np.r_[False,np.linalg.norm(np.diff(arr[:,:2],axis=0),axis=1)>180]
    jumps|=np.r_[False,abs(np.diff(arr[:,5]))>.5]
    for shift in (-2,-1,0,1,2):derivative[np.roll(jumps,shift)]=0
    derivative[:,:2]=np.clip(derivative[:,:2],-3100,3100)
    derivative[:,2:4]=np.clip(derivative[:,2:4],-1300,1300)
    derivative[:,4]=np.clip(derivative[:,4],-15,15)
    output=np.concatenate([arr,derivative],axis=1).astype(np.float32)
    anchor=np.interp(times,source_t,data['anchor_scroll']) if 'anchor_scroll' in data else np.zeros(len(times))
    return output,anchor.astype(np.float32)


def simulate(args):
    import cupy as cp
    device=cp.cuda.runtime.getDeviceProperties(0)
    gpu=device['name'].decode() if isinstance(device['name'],bytes) else str(device['name'])
    print('CUDA DEVICE VERIFIED:',gpu,'CuPy',cp.__version__,flush=True)
    data=np.load(args.input)
    seed=np.asarray(data['xy'],np.float32);n=len(seed)
    glyph=data['glyph'];release=np.asarray(data['release'],np.float32)
    group=data['group'];size=data['size'].astype(np.float32)
    birth=np.asarray(data['birth'],np.float32) if 'birth' in data else np.where(group==3,release,args.start).astype(np.float32)
    # Terminal rows are spatially released by contact and actual fluid drag.
    # A distant glyph keeps its printed anchor until the impact reaches it.
    release=release.copy();release[group==0]=np.float32(1e6)
    # The source reservoir reaches x=4700. The padded physical domain keeps
    # the complete source off-screen until advection carries it into the shot.
    margin=np.array([240.,270.],np.float32)
    sim=Fluid(cp,args.grid[0],args.grid[1],args.dx,args.hz,args.viscosity)
    domain=(args.grid[0]*args.dx,args.grid[1]*args.dx)
    assert seed[:,0].max()+margin[0]<domain[0], 'Input reservoir exceeds physical domain'
    xy=seed+margin
    px=cp.asarray(xy[:,0].copy());py=cp.asarray(xy[:,1].copy())
    sx=px.copy();sy=py.copy();vx=cp.zeros(n,cp.float32);vy=vx.copy()
    angle=vx.copy();spin=vx.copy();samplecurl=vx.copy()
    rel=cp.asarray(release)
    release_dose=cp.zeros(n,cp.float32)
    gpu_group=cp.asarray(group.astype(np.int32))
    gather_flags=np.zeros(n,np.int32);gather_ids=np.empty(0,np.int32);reference=None
    if args.gather_reference:
        reference=np.load(args.gather_reference)
        assert len(reference['group'])==n,'Gather reference particle identities changed'
        from glyph_material import visible_glyphs
        shown=visible_glyphs(group)&(group<3)
        baseline=reference['xy'][np.argmin(abs(reference['times']-8.0))]
        capture=shown&(baseline[:,0]>0)&(baseline[:,0]<1920)&(baseline[:,1]>0)&(baseline[:,1]<1080)&(birth<8.)
        gather_flags[shown]=2;gather_flags[capture]=1
        gather_ids=np.flatnonzero(capture).astype(np.int32)
        print('Annular capture of stable visible source IDs:',len(gather_ids),flush=True)
    gpu_gather=cp.asarray(gather_flags)
    rng=np.random.default_rng(5202026)
    tau=cp.asarray((.031+.0010*size+.018*rng.random(n)+.045*(group==2)).astype(np.float32))
    if not args.glyph_footprints:raise ValueError('Pass --glyph-footprints exported by glyph_contact_geometry.py')
    measured=np.load(args.glyph_footprints)
    assert np.array_equal(measured['glyph'],glyph) and np.array_equal(measured['size'],size),'Glyph geometry does not match seed IDs'
    footprint=measured['footprint'].astype(np.float32);guard_skin=float(measured['skin']);skin=0.
    # Printed source grids have no true overlaps. Raster uncertainty is not
    # physical material: only the broad phase retains the 2.5px raster guard.
    gpu_shape=cp.asarray(footprint.ravel());gpu_birth=cp.asarray(birth)
    raster_host=[np.ascontiguousarray(measured[k]) for k in ['raster_ids','raster_offsets','raster_width','raster_height','raster_bits']]
    gpu_raster=tuple(cp.asarray(v) for v in raster_host)
    ink_radius=np.linalg.norm(footprint[:,:2]+guard_skin,axis=1).astype(np.float32)
    radius=cp.asarray(ink_radius)
    band=np.searchsorted(sim.clearance_radii,ink_radius).astype(np.int32)
    assert band.max()<len(sim.clearance_radii),'Glyph exceeds clearance radius cache'
    gpu_band=cp.asarray(band)
    spring_radius=cp.asarray(np.clip(size*.22,4,8).astype(np.float32))
    shapes=[cp.zeros(n,cp.float32) for _ in range(4)]
    shiftx=cp.zeros(n,cp.float32);shifty=cp.zeros(n,cp.float32);penetration=cp.zeros(n,cp.float32);hits=cp.zeros(n,cp.int32)
    contact_diagnostics=[]
    from glyph_material import visible_glyphs
    displayed_ids=visible_glyphs(group)
    recovery_records=[];recovery_distance=cp.zeros(n,cp.float32);recovery_failed=cp.zeros(n,cp.int32);recovery_needs=cp.zeros(n,cp.int32);recovery_total_distance=cp.zeros(n,cp.float32);retry_ids=cp.zeros(n,cp.int32)
    bin_cell=float(math.ceil(2*np.max(np.linalg.norm(footprint[:,:2]+guard_skin,axis=1)+np.linalg.norm(footprint[:,2:],axis=1))));bin_w=math.ceil(domain[0]/bin_cell);bin_h=math.ceil(domain[1]/bin_cell);capacity=256
    bin_counts=cp.zeros(bin_w*bin_h,cp.int32);bin_ids=cp.empty(bin_w*bin_h*capacity,cp.int32)
    repelx=cp.zeros(n,cp.float32);repely=cp.zeros(n,cp.float32)
    steps=round((args.end-args.start)*args.hz)
    timestamps=args.start+np.arange(steps+1,dtype=np.float64)/args.hz
    obstacles,anchors=obstacle_track(args.obstacles,timestamps)
    mask_times=masks=solid_velocity=None
    if args.solid_masks:
        mask_source=np.load(args.solid_masks)
        mask_times=mask_source['times'];masks=mask_source['solid_mask']
        solid_velocity=mask_source['solid_velocity']
        print('Actual articulated masks loaded',masks.shape,flush=True)
    last_mask=-1
    stride=round(args.hz/args.fps);assert abs(args.hz/stride-args.fps)<1e-6
    output={k:[] for k in ('times','xy','velocity','angle','curl','attached')}
    start=time.perf_counter()
    for j,t in enumerate(timestamps):
        if reference is not None and j==round((8.05-args.start)*args.hz):
            # Resume from the exact previously rendered state immediately
            # before force engagement, retaining momentum and angular spin.
            ri=int(np.searchsorted(reference['times'],8.05)-1)
            hp=reference['xy'][ri]+margin;hv=reference['velocity'][ri]
            ha=reference['angle'];ht=reference['times']
            hs=(ha[ri+1]-ha[ri-1])/(ht[ri+1]-ht[ri-1])*np.float32(math.pi/180)
            px.set(np.ascontiguousarray(hp[:,0]));py.set(np.ascontiguousarray(hp[:,1]))
            vx.set(np.ascontiguousarray(hv[:,0]));vy.set(np.ascontiguousarray(hv[:,1]))
            angle.set(np.ascontiguousarray(ha[ri]));spin.set(np.ascontiguousarray(hs))
            cp.cuda.get_current_stream().synchronize()
            print('Resumed exact reference particle checkpoint',float(ht[ri]),flush=True)
        if mask_times is not None:
            index=int(np.clip(np.searchsorted(mask_times,t+1e-4,side='right')-1,0,len(mask_times)-1))
            if index!=last_mask:
                sim.update_solid_mask(masks[index],solid_velocity[index]);last_mask=index
        diagnostic=(j%max(1,round(args.hz*.5))==0)
        sim.anchor_kernel(((n+255)//256,),(256,),
                          (px,py,sx,sy,rel,gpu_group,np.int32(n),np.float32(t),np.float32(anchors[j])))
        if j > 0:
            sim.step(float(t),obstacles[j],margin,diagnostic)
            bin_counts.fill(0)
            sim.bin_kernel(((n+255)//256,),(256,),
                           (px,py,rel,gpu_birth,gpu_group,bin_counts,bin_ids,np.int32(n),np.int32(bin_w),np.int32(bin_h),np.int32(capacity),np.float32(bin_cell),np.float32(t)))
            sim.contact_kernel(((n+255)//256,),(256,),
                               (px,py,vx,vy,spring_radius,rel,gpu_group,bin_counts,bin_ids,repelx,repely,
                                np.int32(n),np.int32(bin_w),np.int32(bin_h),np.int32(capacity),np.float32(bin_cell),np.float32(t),np.float32(domain[0]),np.float32(domain[1])))
            sim.release_kernel(((n+255)//256,),(256,),
                               (px,py,rel,release_dose,gpu_group,radius,repelx,repely,
                                sim.u,sim.v,sim.sdf,np.int32(n),np.int32(sim.w),np.int32(sim.h),
                                np.int32(sim.maskw),np.int32(sim.maskh),np.int32(sim.usemask),
                                np.float32(args.dx),np.float32(margin[0]),np.float32(margin[1]),np.float32(sim.dt),np.float32(t)))
            sim.tracers(((n+255)//256,),(256,),
                        (px,py,vx,vy,angle,spin,samplecurl,sx,sy,rel,tau,gpu_group,repelx,repely,gpu_gather,
                         sim.u,sim.v,sim.curl,sim.obs,sim.sdf,sim.normalx,sim.normaly,sim.solidu,sim.solidv,
                         np.int32(sim.maskw),np.int32(sim.maskh),np.int32(sim.usemask),np.float32(margin[0]),np.float32(margin[1]),
                         np.int32(n),np.int32(sim.w),np.int32(sim.h),
                         np.float32(args.dx),np.float32(sim.dt),np.float32(t),np.float32(anchors[j]),
                         np.float32(domain[0]),np.float32(domain[1])))
        # Resolve solid clearance and oriented glyph contacts at every integration
        # step. At recorded frames, converge further before a camera can see it.
        iterations=args.contact_iterations
        if j%stride==0:iterations=args.output_contact_iterations
        for iteration in range(iterations):
            if sim.usemask:
                sim.project_solids(((n+255)//256,),(256,),
                    (px,py,vx,vy,spin,angle,gpu_shape,rel,gpu_birth,gpu_band,
                     sim.clearance_sdf,sim.clearance_normalx,sim.clearance_normaly,sim.contact_solidu,sim.contact_solidv,np.int32(n),
                     np.int32(sim.maskw),np.int32(sim.maskh),np.float32(args.dx),
                     np.float32(margin[0]),np.float32(margin[1]),np.float32(t),np.int32(iteration==0 and j>0)))
            sim.prepare_shapes(((n+255)//256,),(256,),(px,py,angle,gpu_shape,*shapes,np.int32(n)))
            bin_counts.fill(0)
            sim.bin_kernel(((n+255)//256,),(256,),
                       (px,py,rel,gpu_birth,gpu_group,bin_counts,bin_ids,np.int32(n),np.int32(bin_w),np.int32(bin_h),np.int32(capacity),np.float32(bin_cell),np.float32(t)))
            if t<16.5:
                for color in range(9):
                    sim.solve_contacts_colored(((bin_w*bin_h+127)//128,),(128,),
                      (px,py,vx,vy,spin,*shapes,gpu_shape,rel,gpu_birth,gpu_group,bin_counts,bin_ids,
                       np.int32(n),np.int32(bin_w),np.int32(bin_h),np.int32(capacity),np.float32(bin_cell),np.float32(t),
                       np.float32(skin),np.int32((color+j)%9),np.float32(1/sim.dt),np.float32(.04 if iteration==0 and j>0 else 0.),
                       gpu_band,sim.clearance_sdf,sim.clearance_normalx,sim.clearance_normaly,np.int32(sim.maskw),np.int32(sim.maskh),np.float32(args.dx),np.float32(margin[0]),np.float32(margin[1])))
            else:
                # The dissolving 9px worker mesh has thousands of tiny bodies.
                # Parallel Jacobi propagates that contact network efficiently;
                # both methods use the same OBB constraints and final audit.
                sim.project_contacts(((n+255)//256,),(256,),
                    (px,py,*shapes,gpu_shape,rel,gpu_birth,gpu_group,bin_counts,bin_ids,shiftx,shifty,penetration,hits,
                     np.int32(n),np.int32(bin_w),np.int32(bin_h),np.int32(capacity),np.float32(bin_cell),np.float32(t),np.float32(skin),
                     gpu_band,sim.clearance_sdf,sim.clearance_normalx,sim.clearance_normaly,np.int32(sim.maskw),np.int32(sim.maskh),np.float32(args.dx),np.float32(margin[0]),np.float32(margin[1])))
                sim.apply_contacts(((n+255)//256,),(256,),
                    (px,py,vx,vy,spin,shiftx,shifty,hits,np.int32(n),np.float32(1/sim.dt),np.float32(.04 if iteration==0 and j>0 else 0.)))
            if j%stride==0 and (iteration%8==7 or iteration==iterations-1):
                # Independent SAT diagnostic after the complete position update.
                sim.prepare_shapes(((n+255)//256,),(256,),(px,py,angle,gpu_shape,*shapes,np.int32(n)))
                bin_counts.fill(0)
                sim.bin_kernel(((n+255)//256,),(256,),(px,py,rel,gpu_birth,gpu_group,bin_counts,bin_ids,np.int32(n),np.int32(bin_w),np.int32(bin_h),np.int32(capacity),np.float32(bin_cell),np.float32(t)))
                sim.project_contacts(((n+255)//256,),(256,),
                    (px,py,*shapes,gpu_shape,rel,gpu_birth,gpu_group,bin_counts,bin_ids,shiftx,shifty,penetration,hits,
                     np.int32(n),np.int32(bin_w),np.int32(bin_h),np.int32(capacity),np.float32(bin_cell),np.float32(t),np.float32(skin),
                     gpu_band,sim.clearance_sdf,sim.clearance_normalx,sim.clearance_normaly,np.int32(sim.maskw),np.int32(sim.maskh),np.float32(args.dx),np.float32(margin[0]),np.float32(margin[1])))
                if iteration>=args.contact_iterations and float(cp.max(penetration).get())<max(.05,2*skin-.5):break
        # Bounded feasibility recovery handles nonlinear corners after local
        # contact iterations. It runs at EVERY 120Hz step, updates the actual
        # integrator state and momentum, and records every correction.
        recovery_total_distance.fill(0)
        for recovery_pass in range(4):
            sim.prepare_shapes(((n+255)//256,),(256,),(px,py,angle,gpu_shape,*shapes,np.int32(n)))
            bin_counts.fill(0)
            sim.bin_kernel(((n+255)//256,),(256,),
                           (px,py,rel,gpu_birth,gpu_group,bin_counts,bin_ids,np.int32(n),np.int32(bin_w),np.int32(bin_h),np.int32(capacity),np.float32(bin_cell),np.float32(t)))
            if recovery_pass>0:retry_ids[:]=recovery_failed
            sim.flag_recovery(((n+255)//256,),(256,),
                (px,py,*shapes,gpu_shape,rel,gpu_birth,bin_counts,bin_ids,
                 np.int32(n),np.int32(bin_w),np.int32(bin_h),np.int32(capacity),np.float32(bin_cell),np.float32(t),np.float32(guard_skin),
                 gpu_band,sim.clearance_sdf,np.int32(sim.maskw),np.int32(sim.maskh),np.float32(args.dx),np.float32(margin[0]),np.float32(margin[1]),
                 *gpu_raster,recovery_needs))
            if recovery_pass>0:recovery_needs*=retry_ids
            sim.recover_contacts((1,),(32,),
                (px,py,vx,vy,spin,*shapes,gpu_shape,rel,gpu_birth,bin_counts,bin_ids,
                 np.int32(n),np.int32(bin_w),np.int32(bin_h),np.int32(capacity),np.float32(bin_cell),np.float32(t),np.float32(guard_skin),
                 gpu_band,sim.clearance_sdf,np.int32(sim.maskw),np.int32(sim.maskh),np.float32(args.dx),np.float32(margin[0]),np.float32(margin[1]),np.float32(1/sim.dt),
                 recovery_distance,recovery_failed,np.int32(j%2),recovery_needs,*gpu_raster))
            recovery_total_distance+=recovery_distance
            if not bool(cp.any(recovery_failed).get()):break
        distances=cp.asnumpy(recovery_total_distance);positive=distances[distances>0]
        recovery_screen=cp.asnumpy((px>margin[0]-60)&(px<margin[0]+1980)&(py>margin[1]-60)&(py<margin[1]+1140))&displayed_ids&(birth<=t)
        visible_distances=distances[recovery_screen&(distances>0)]
        recovery_records.append((float(t),len(positive),float(np.percentile(positive,95)) if len(positive) else 0.,float(positive.max()) if len(positive) else 0.,int(cp.sum(recovery_failed).get()),len(visible_distances),float(np.percentile(visible_distances,95)) if len(visible_distances) else 0.,float(visible_distances.max()) if len(visible_distances) else 0.,int(np.sum(visible_distances>=8)),recovery_pass+1))
        if j%stride==0:
            sim.prepare_shapes(((n+255)//256,),(256,),(px,py,angle,gpu_shape,*shapes,np.int32(n)))
            bin_counts.fill(0)
            sim.bin_kernel(((n+255)//256,),(256,),(px,py,rel,gpu_birth,gpu_group,bin_counts,bin_ids,np.int32(n),np.int32(bin_w),np.int32(bin_h),np.int32(capacity),np.float32(bin_cell),np.float32(t)))
            sim.project_contacts(((n+255)//256,),(256,),
                (px,py,*shapes,gpu_shape,rel,gpu_birth,gpu_group,bin_counts,bin_ids,shiftx,shifty,penetration,hits,
                 np.int32(n),np.int32(bin_w),np.int32(bin_h),np.int32(capacity),np.float32(bin_cell),np.float32(t),np.float32(skin),
                 gpu_band,sim.clearance_sdf,sim.clearance_normalx,sim.clearance_normaly,np.int32(sim.maskw),np.int32(sim.maskh),np.float32(args.dx),np.float32(margin[0]),np.float32(margin[1])))
            active=t>=gpu_birth
            screen=active&(px>margin[0]-60)&(px<margin[0]+1980)&(py>margin[1]-60)&(py<margin[1]+1140)
            contact_diagnostics.append((float(t),float(cp.max(cp.where(screen,penetration,0)).get()),int(cp.sum(screen&(penetration>2*skin)).get()),int(cp.max(bin_counts).get()),iteration+1))
            output['times'].append(t)
            output['xy'].append(cp.asnumpy(cp.stack([px-margin[0],py-margin[1]],axis=-1)))
            output['velocity'].append(cp.asnumpy(cp.stack([vx,vy],axis=-1)))
            output['angle'].append(cp.asnumpy(angle));output['curl'].append(cp.asnumpy(samplecurl))
            output['attached'].append(cp.asnumpy(t<rel))
            if any(abs(t-mark)<.25/args.hz for mark in args.checkpoint_times):
                checkpoint=Path(args.output).with_name(Path(args.output).stem+f'.at-{t:.2f}.npz')
                partial={k:np.asarray(v,dtype=np.bool_ if k=='attached' else np.float32) for k,v in output.items()}
                np.savez_compressed(checkpoint,**partial,glyph=glyph,group=group,size=size,seed_xy=seed,
                                    birth=birth,release=cp.asnumpy(rel),contact_diagnostics=np.asarray(contact_diagnostics,np.float32))
                print('QA CHECKPOINT',checkpoint,'contact',contact_diagnostics[-1],flush=True)
        if diagnostic and j > 0:
            row=sim.diag[-1]
            print(f't={t:.3f} / {args.end:.2f} | div RMS {row[1]:.5g} -> {row[2]:.5g} | max speed {row[3]:.1f} px/s | {time.perf_counter()-start:.1f}s | contacts {contact_diagnostics[-1] if contact_diagnostics else None}',flush=True)
    cp.cuda.Stream.null.synchronize()
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    packed={k:np.asarray(v,dtype=np.bool_ if k=='attached' else np.float32) for k,v in output.items()}
    release=cp.asnumpy(rel)
    if reference is not None:
        # The unchanged terminal-covered checkpoint resumes the already
        # approved later GPU simulation. Keep the pre-impulse film identical.
        restore=(packed['times']<8.05)|(packed['times']>=10.)
        for key in ('xy','velocity','angle','curl','attached'):
            packed[key][restore]=reference[key][restore]
        release=reference['release']
        handoff_i=int(np.argmin(abs(packed['times']-8.5)))
        omega=(packed['angle'][handoff_i+1]-packed['angle'][handoff_i-1])/(packed['times'][handoff_i+1]-packed['times'][handoff_i-1])
        printed_size=size[gather_ids]
        handoff_path=Path(args.handoff_output or out.with_name('gather-handoff.npz'))
        np.savez_compressed(handoff_path,time=np.float32(8.5),source_ids=gather_ids,
                            xy=packed['xy'][handoff_i,gather_ids],velocity=packed['velocity'][handoff_i,gather_ids],
                            angle=packed['angle'][handoff_i,gather_ids],angular_velocity=omega[gather_ids],
                            glyph=glyph[gather_ids],size=printed_size.astype(np.float32),raw_size=size[gather_ids],
                            group=group[gather_ids],center=np.array([960,540],np.float32))
        r=np.linalg.norm(packed['xy'][handoff_i,gather_ids]-[960,540],axis=1)
        print('HANDOFF',handoff_path,'radius percentiles',np.percentile(r,[0,10,50,90,100]),flush=True)
    metadata=dict(gpu=gpu,cupy=cp.__version__,solver='RK2 semi-Lagrangian + spectral pressure projection + viscosity + vorticity confinement + Brinkman moving boundaries + inertial tracers',
                  grid=args.grid,domain_px=domain,margin_px=margin.tolist(),dt=sim.dt,viscosity_px2_s=args.viscosity,
                  simulated_frames=len(packed['times']),particles=n,seconds_wall=time.perf_counter()-start)
    metadata['state_time_alignment']='initial seed at start; integrate exactly one dt for each subsequent timestamp'
    metadata['actual_silhouette_obstacles']=bool(args.solid_masks)
    metadata['solid_mask_samples']=dict(path=str(args.solid_masks),frames=len(mask_times),hz=float(1/np.median(np.diff(mask_times)))) if mask_times is not None else None
    metadata['input_preparation']=json.loads(str(data['preparation_metadata'])) if 'preparation_metadata' in data else None
    metadata['active_body_counts']={str(g):int(np.sum((group==g)&(birth<=args.end))) for g in np.unique(group)}
    metadata['cluster_sequence']=dict(version=5,missing_rank=23,start=7.30,dock=8.56,
                                     network=10.20,deploy=12.20,late_rank_entry=14.42,late_rank_dock=15.32,
                                     unfold=16.10,handoff=16.12,
                                     legacy_annular_forces=False)
    metadata['opening_impact']=dict(visible_body='RUN rigid cursor',source='opening_impact.py',
                                  anchored_contact_targets=True,release='solid contact, particle impulse, or accumulated local fluid drag',
                                  free_fan_impulses=False)
    metadata['finite_size_contacts']=dict(model='all-group oriented Menlo ink rectangles, wall-tangent active-set Gauss-Seidel and parallel Jacobi position constraints, reinitialized configuration-space solid clearance and surface friction',skin_px=skin,raster_broad_phase_guard_px=guard_skin,hash_cell_px=bin_cell,hash_capacity=capacity,same_group_only=False,particle_boundaries='open outflow; no wrap/reappearance',contact_iterations=args.contact_iterations,output_contact_iterations=args.output_contact_iterations,geometry_source=str(args.glyph_footprints),solid_clearance_radii_px=sim.clearance_radii.tolist(),diagnostic_columns=['time','max_screen_constraint_penetration_px','screen_glyphs_over_ink_skin','max_bin_occupancy','iterations'])
    metadata['contact_field_exterior']=dict(padding_cells=16,padding_px=16*args.dx,boundary='free exterior; padded distances cropped to original mask coordinates; nonperiodic normal differences')
    metadata['contact_velocity_response']='one moving-wall reflection/friction and one pair-impulse sweep per120Hz step; subsequent constraint sweeps update positions only'
    metadata['nonlinear_contact_recovery']=dict(method='bounded nearest feasible position projection using exact rotated Menlo raster contacts, into persistent integrator state at every substep',max_search_px=192,velocity_response=.08,max_recovery_passes=4,direction_search='warp-parallel32 candidates, lowest-valid direction, sequential body commits',raster_rounding='float32 camera coordinates followed by double-precision ties-to-even paste rounding',columns=['time','corrections','p95_px','max_px','unresolved','visible_corrections','visible_p95_px','visible_max_px','visible_corrections_ge_8px','recovery_passes'])
    metadata['annular_gather']=dict(enabled=bool(args.gather_reference),start=8.05,handoff=8.5,source_ids=len(gather_ids),reference_checkpoint_resume=10.)
    np.savez_compressed(out,**packed,glyph=glyph,group=group,size=size,seed_xy=seed,birth=birth,
                        release=release,gather_ids=gather_ids,recovery_diagnostics=np.asarray(recovery_records,np.float32),contact_diagnostics=np.asarray(contact_diagnostics,np.float32),diagnostics=np.asarray(sim.diag,np.float32),metadata=np.array(json.dumps(metadata)))
    out.with_suffix('.json').write_text(json.dumps(metadata,indent=2))
    print('SAVED',out,out.stat().st_size,'bytes',json.dumps(metadata),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--input',required=True);p.add_argument('--obstacles');p.add_argument('--output',required=True)
    p.add_argument('--solid-masks')
    p.add_argument('--glyph-footprints',required=True)
    p.add_argument('--contact-iterations',type=int,default=12)
    p.add_argument('--checkpoint-times',type=float,nargs='*',default=[10.,16.,17.5])
    p.add_argument('--output-contact-iterations',type=int,default=64)
    p.add_argument('--gather-reference');p.add_argument('--handoff-output')
    p.add_argument('--grid',type=int,nargs=2,default=[1536,432]);p.add_argument('--dx',type=float,default=3.75)
    p.add_argument('--hz',type=int,default=120);p.add_argument('--fps',type=int,default=24)
    p.add_argument('--start',type=float,default=3.0);p.add_argument('--end',type=float,default=9.75)
    p.add_argument('--viscosity',type=float,default=11.)
    simulate(p.parse_args())
