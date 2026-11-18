/* Adapter to Paiva et al.'s Apache-2.0 FFT implementation.
 * Evaluates periodic-neighbor pairwise halfspaces and toroidal NN diagnostics.
 * Grid wrap error is bounded separately using marginal Chernoff tails.
 */
#define main original_paiva_main
#include "vendor/joint_distribution_2d.c"
#undef main
#include <time.h>

static int td2(int x,int y){ x=mod_centered(x,3329); y=mod_centered(y,3329); return x*x+y*y; }
static void output(mpfr_t x){mpfr_out_str(stdout,10,35,x,MPFR_RNDN);}
int main(int argc, char **argv){
 if(argc!=2){fprintf(stderr,"Expected a directions file\n");return 2;}
 int nh[2],H[2][64][3]; FILE *hf=fopen(argv[1],"r");
 if(!hf)return 2;
 for(int c=0;c<2;c++){if(fscanf(hf,"%d",&nh[c])!=1 || nh[c]<1 || nh[c]>64)return 2;
 for(int i=0;i<nh[c];i++)if(fscanf(hf,"%d%d%d",&H[c][i][0],&H[c][i][1],&H[c][i][2])!=3)return 2;}
 fclose(hf);
 double start=omp_get_wtime();
 init_fft_mpc();
 prob_dist_1d_t a,cu,cv,b,e;
 prob_dist_1d_init_as_centered_binomial(&a,2);
 prob_dist_1d_init_as_compress_decompress_error(&cu,11,3329);
 prob_dist_1d_init_as_compress_decompress_error(&cv,5,3329);
 prob_dist_1d_init_as_sum(&b,&a,&cu);
 prob_dist_1d_init_as_sum(&e,&a,&cv);
 mpc_matrix_t F,G;
 fprintf(stderr,"BASE1 %.2fs\n",omp_get_wtime()-start);
 get_base_2x2_distribution_for_self_convolution(&F,&a,&a);
 fft2(&F); self_convolution_in_fft_domain(&F,512);
 fprintf(stderr,"BASE2 %.2fs\n",omp_get_wtime()-start);
 get_base_2x2_distribution_for_self_convolution(&G,&b,&a);
 fft2(&G); self_convolution_in_fft_domain(&G,512);
 #pragma omp parallel for schedule(static)
 for(int i=0;i<N_SUPPORT_DIST_2D;i++)for(int j=0;j<N_SUPPORT_DIST_2D;j++)mpc_mul(F.m[i][j],F.m[i][j],G.m[i][j],MPC_RNDNN);
 mpc_matrix_free(&G);
 mpc_t *E=malloc(N_SUPPORT_DIST_2D*sizeof(mpc_t));
 for(int i=0;i<N_SUPPORT_DIST_2D;i++){mpc_init2(E[i],N_PRECISION_BITS);mpc_set_ui(E[i],0,MPC_RNDNN);}
 for(int v=e.min_value;v<=e.max_value;v++)mpc_set_fr(E[MOD(v,N_SUPPORT_DIST_2D)],e.d[prob_dist_1d_get_idx_for_value(&e,v)],MPC_RNDNN);
 fft(E,N_SUPPORT_DIST_2D);
 #pragma omp parallel for schedule(static)
 for(int i=0;i<N_SUPPORT_DIST_2D;i++)for(int j=0;j<N_SUPPORT_DIST_2D;j++){
  mpc_mul(F.m[i][j],F.m[i][j],E[i],MPC_RNDNN);mpc_mul(F.m[i][j],F.m[i][j],E[j],MPC_RNDNN);
 }
 for(int i=0;i<N_SUPPORT_DIST_2D;i++)mpc_clear(E[i]);free(E);
 fprintf(stderr,"INVERSE %.2fs\n",omp_get_wtime()-start);ifft2(&F);
 const int ncase=2;
 int C[2][4][2]={{{0,0},{446,1664},{1664,2883},{2110,1218}},{{0,0},{446,1664},{1664,446},{2110,2110}}};
 const char *names[]={"ours_gamma446","previous_gamma446"};
 int nt=omp_get_max_threads();
 mpfr_t sums[nt][2][5],mass[nt],negative[nt],imag[nt];
 for(int t=0;t<nt;t++){
  mpfr_inits2(N_PRECISION_BITS,mass[t],negative[t],imag[t],(mpfr_ptr)0);
  mpfr_set_zero(mass[t],0);mpfr_set_zero(negative[t],0);mpfr_set_zero(imag[t],0);
  for(int c=0;c<ncase;c++)for(int r=0;r<5;r++){mpfr_init2(sums[t][c][r],N_PRECISION_BITS);mpfr_set_zero(sums[t][c][r],0);}
 }
 fprintf(stderr,"DECODE %.2fs\n",omp_get_wtime()-start);
 #pragma omp parallel for schedule(static)
 for(int i=0;i<N_SUPPORT_DIST_2D;i++){
  int t=omp_get_thread_num(),x=center_mod(i,N_SUPPORT_DIST_2D);
  for(int j=0;j<N_SUPPORT_DIST_2D;j++){
   int y=center_mod(j,N_SUPPORT_DIST_2D);
   mpfr_ptr p=mpc_realref(F.m[i][j]);
   mpfr_add(mass[t],mass[t],p,MPFR_RNDN);
   if(mpfr_sgn(p)<0)mpfr_sub(negative[t],negative[t],p,MPFR_RNDN);
   mpfr_t ai;mpfr_init2(ai,N_PRECISION_BITS);mpfr_abs(ai,mpc_imagref(F.m[i][j]),MPFR_RNDN);mpfr_add(imag[t],imag[t],ai,MPFR_RNDN);mpfr_clear(ai);
   int own=td2(x,y);
   for(int c=0;c<ncase;c++){
    int cnt[5]={0};
    for(int i=0;i<nh[c];i++){
     long hx=H[c][i][0],hy=H[c][i][1];
     if(2*(hx*x+hy*y)>=hx*hx+hy*hy)cnt[4]+=H[c][i][2];
    }
    for(int sent=0;sent<4;sent++){
     int le=0,lt=0,det=0;
     for(int comp=0;comp<4;comp++)if(comp!=sent){
      int d=td2(x+C[c][sent][0]-C[c][comp][0],y+C[c][sent][1]-C[c][comp][1]);
      le+=(d<=own);lt+=(d<own);det|=(d<own || (d==own&&comp<sent));
     }
     cnt[0]+=(le>0);cnt[1]+=(lt>0);cnt[2]+=det;cnt[3]+=le;
    }
    for(int r=0;r<5;r++)if(cnt[r]){
     mpfr_t tmp;mpfr_init2(tmp,N_PRECISION_BITS);mpfr_mul_ui(tmp,p,cnt[r],MPFR_RNDN);mpfr_add(sums[t][c][r],sums[t][c][r],tmp,MPFR_RNDN);mpfr_clear(tmp);
    }
   }
  }
 }
 for(int t=1;t<nt;t++){
  mpfr_add(mass[0],mass[0],mass[t],MPFR_RNDN);mpfr_add(negative[0],negative[0],negative[t],MPFR_RNDN);mpfr_add(imag[0],imag[0],imag[t],MPFR_RNDN);
  for(int c=0;c<ncase;c++)for(int r=0;r<5;r++)mpfr_add(sums[0][c][r],sums[0][c][r],sums[t][c][r],MPFR_RNDN);
 }
 printf("{\"precision_bits\":%d,\"grid\":%d,\"mass\":",N_PRECISION_BITS,N_SUPPORT_DIST_2D);output(mass[0]);
 printf(",\"negative_mass\":");output(negative[0]);printf(",\"imaginary_l1\":");output(imag[0]);printf(",\"cases\":[");
 const char *rules[]={"ties_error","strict_error","deterministic_first","pairwise_ties","voronoi_pairwise"};
 for(int c=0;c<ncase;c++){
  if(c)printf(",");printf("{\"name\":\"%s\"",names[c]);
  for(int r=0;r<5;r++){
   mpfr_mul_ui(sums[0][c][r],sums[0][c][r],32,MPFR_RNDN);
   printf(",\"%s\":",rules[r]);output(sums[0][c][r]);
   mpfr_t l;mpfr_init2(l,N_PRECISION_BITS);mpfr_log2(l,sums[0][c][r],MPFR_RNDN);
   printf(",\"log2_%s\":%.14f",rules[r],mpfr_get_d(l,MPFR_RNDN));mpfr_clear(l);
  }printf("}");
 }
 printf("],\"elapsed_seconds\":%.3f}\n",omp_get_wtime()-start);fflush(stdout);
 mpc_matrix_free(&F);clear_fft_mpc();return 0;
}
