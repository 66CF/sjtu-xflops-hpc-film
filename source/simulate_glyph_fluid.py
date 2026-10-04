#!/usr/bin/env python3
"""CUDA incompressible fluid and inertial ASCII tracers for the Xflops film.

This advances a velocity field, not precomputed paths or target-position lerps:
RK2 semi-Lagrangian velocity transport, viscous diffusion, vorticity confinement,
moving-solid Brinkman coupling, a Fourier pressure Poisson projection, followed
by inertial particle drag, contact impulses and angular response to fluid curl.

The padded periodic physical domain wraps only well outside the film aperture.
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
                float bu=sample(solidu,mx,my,maskw,maskh),bv=sample(solidv,mx,my,maskw,maskh);
                float relx=vx[i]-bu,rely=vy[i]-bv;
                float vn=relx*nx+rely*ny;
                if(vn<0){vx[i]-=1.06f*vn*nx;vy[i]-=1.06f*vn*ny;}
                float tangent=relx*(-ny)+rely*nx;
                vx[i]+=.09f*tangent*ny;vy[i]-=.09f*tangent*nx;
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
    if(!(time>=8.05f && time<=9.75f && gather[i]==2)) {
        px[i]-=floorf(px[i]/domainw)*domainw;
        py[i]-=floorf(py[i]/domainh)*domainh;
    }
}
__global__ void boundary_seed(const unsigned char* mask,float* sx,float* sy,int w,int h) {
    int id=blockDim.x*blockIdx.x+threadIdx.x;if(id>=w*h)return;
    int x=id%w,y=id/w;unsigned char m=mask[id];
    bool edge=(x>0 && mask[id-1]!=m)||(x<w-1 && mask[id+1]!=m)||
              (y>0 && mask[id-w]!=m)||(y<h-1 && mask[id+w]!=m);
    sx[id]=edge?x:-10000;sy[id]=edge?y:-10000;
}
__global__ void particle_bins(const float* px,const float* py,const float* release,const int* group,
                             int* count,int* ids,int n,int bw,int bh,int capacity,
                             float cell,float time) {
    int i=blockDim.x*blockIdx.x+threadIdx.x;if(i>=n||(time<release[i] && group[i]!=0))return;
    int x=((int)floorf(px[i]/cell))%bw,y=((int)floorf(py[i]/cell))%bh;
    x=(x+bw)%bw;y=(y+bh)%bh;int b=y*bw+x;
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
            int j=ids[b*capacity+k];if(j==i||group[j]!=group[i])continue;
            float dx=px[i]-px[j],dy=py[i]-py[j];
            dx-=nearbyintf(dx/domainw)*domainw;dy-=nearbyintf(dy/domainh)*domainh;
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
        module=cp.RawModule(code=CUDA,options=('--std=c++11',),name_expressions=('advect','tracers','boundary_seed','jump_flood','particle_bins','contact_forces','anchor_terminal','release_terminal'))
        self.advect=module.get_function('advect');self.tracers=module.get_function('tracers')
        self.obs=cp.zeros(11,cp.float32)
        self.curl=cp.zeros_like(self.u)
        self.seed_kernel=module.get_function('boundary_seed');self.jump_kernel=module.get_function('jump_flood')
        self.bin_kernel=module.get_function('particle_bins');self.contact_kernel=module.get_function('contact_forces')
        self.anchor_kernel=module.get_function('anchor_terminal');self.release_kernel=module.get_function('release_terminal')
        self.maskw=512;self.maskh=288
        self.solidmask=cp.zeros((self.maskh,self.maskw),cp.uint8)
        self.sdf=cp.full((self.maskh,self.maskw),1e4,cp.float32)
        self.normalx=cp.zeros_like(self.sdf);self.normaly=cp.zeros_like(self.sdf)
        self.solidu=cp.zeros_like(self.sdf);self.solidv=cp.zeros_like(self.sdf)
        self.usemask=False
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
            self.sdf.fill(1e4);self.normalx.fill(0);self.normaly.fill(0);return
        sx=cp.empty_like(self.sdf);sy=cp.empty_like(self.sdf);tx=cp.empty_like(self.sdf);ty=cp.empty_like(self.sdf)
        blocks=((self.maskw*self.maskh+255)//256,)
        self.seed_kernel(blocks,(256,),(self.solidmask,sx,sy,np.int32(self.maskw),np.int32(self.maskh)))
        for step in [256,128,64,32,16,8,4,2,1,1]:
            self.jump_kernel(blocks,(256,),(sx,sy,tx,ty,np.int32(self.maskw),np.int32(self.maskh),np.int32(step)))
            sx,tx=tx,sx;sy,ty=ty,sy
        yy,xx=cp.indices((self.maskh,self.maskw),dtype=cp.float32)
        distance=cp.sqrt((xx-sx)**2+(yy-sy)**2)*self.dx
        self.sdf=(distance+.5*self.dx)*cp.where(self.solidmask>0,-1.,1.).astype(cp.float32)
        self.normalx=(cp.roll(self.sdf,-1,1)-cp.roll(self.sdf,1,1))/(2*self.dx)
        self.normaly=(cp.roll(self.sdf,-1,0)-cp.roll(self.sdf,1,0))/(2*self.dx)
        norm=cp.sqrt(self.normalx*self.normalx+self.normaly*self.normaly+1e-8)
        self.normalx/=norm;self.normaly/=norm

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
        # The documented editorial impulses are real force terms, rather than
        # drawing-position interpolation: one diagonal jet, then a center fan.
        diagonal=math.exp(-((t-7.71)/.15)**2)
        if diagonal>.002:
            self.u+=dt*(-1450)*diagonal;self.v+=dt*(-740)*diagonal
        knot=float(smooth((t-8.10)/.27)*(1-smooth((t-9.35)/.35)))
        if knot>.001:
            ox=xx-960;oy=yy-540;r2=ox*ox+oy*oy
            fall=cp.exp(-r2/(680**2))
            self.u+=dt*knot*(-oy)*fall*2.1
            self.v+=dt*knot*ox*fall*2.1
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
    radius=cp.asarray(np.clip(size*.22,4,8).astype(np.float32))
    bin_cell=24.;bin_w=math.ceil(domain[0]/bin_cell);bin_h=math.ceil(domain[1]/bin_cell);capacity=128
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
        sim.step(float(t),obstacles[j],margin,diagnostic)
        sim.anchor_kernel(((n+255)//256,),(256,),
                          (px,py,sx,sy,rel,gpu_group,np.int32(n),np.float32(t),np.float32(anchors[j])))
        bin_counts.fill(0)
        sim.bin_kernel(((n+255)//256,),(256,),
                       (px,py,rel,gpu_group,bin_counts,bin_ids,np.int32(n),np.int32(bin_w),np.int32(bin_h),np.int32(capacity),np.float32(bin_cell),np.float32(t)))
        sim.contact_kernel(((n+255)//256,),(256,),
                           (px,py,vx,vy,radius,rel,gpu_group,bin_counts,bin_ids,repelx,repely,
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
        if j%stride==0:
            output['times'].append(t)
            output['xy'].append(cp.asnumpy(cp.stack([px-margin[0],py-margin[1]],axis=-1)))
            output['velocity'].append(cp.asnumpy(cp.stack([vx,vy],axis=-1)))
            output['angle'].append(cp.asnumpy(angle));output['curl'].append(cp.asnumpy(samplecurl))
            output['attached'].append(cp.asnumpy(t<rel))
        if diagnostic:
            row=sim.diag[-1]
            print(f't={t:.3f} / {args.end:.2f} | div RMS {row[1]:.5g} -> {row[2]:.5g} | max speed {row[3]:.1f} px/s | {time.perf_counter()-start:.1f}s',flush=True)
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
    metadata['actual_silhouette_obstacles']=bool(args.solid_masks)
    metadata['opening_impact']=dict(visible_body='RUN rigid cursor',source='opening_impact.py',
                                  anchored_contact_targets=True,release='solid contact, particle impulse, or accumulated local fluid drag',
                                  free_fan_impulses=False)
    metadata['finite_size_contacts']={'radius_px':[4,8],'spring_s2':600,'damping_s1':14,'hash_cell_px':24,'same_group_only':True}
    metadata['annular_gather']=dict(enabled=bool(args.gather_reference),start=8.05,handoff=8.5,source_ids=len(gather_ids),reference_checkpoint_resume=10.)
    np.savez_compressed(out,**packed,glyph=glyph,group=group,size=size,seed_xy=seed,birth=birth,
                        release=release,gather_ids=gather_ids,diagnostics=np.asarray(sim.diag,np.float32),metadata=np.array(json.dumps(metadata)))
    out.with_suffix('.json').write_text(json.dumps(metadata,indent=2))
    print('SAVED',out,out.stat().st_size,'bytes',json.dumps(metadata),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--input',required=True);p.add_argument('--obstacles');p.add_argument('--output',required=True)
    p.add_argument('--solid-masks')
    p.add_argument('--gather-reference');p.add_argument('--handoff-output')
    p.add_argument('--grid',type=int,nargs=2,default=[1536,432]);p.add_argument('--dx',type=float,default=3.75)
    p.add_argument('--hz',type=int,default=120);p.add_argument('--fps',type=int,default=24)
    p.add_argument('--start',type=float,default=3.0);p.add_argument('--end',type=float,default=9.75)
    p.add_argument('--viscosity',type=float,default=11.)
    simulate(p.parse_args())
