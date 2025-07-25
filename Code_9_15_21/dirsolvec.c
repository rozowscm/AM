#include "mex.h"
#include <math.h>

/*
% DIGAMMA Computes the digamma (psi) function for positive arguments
% USAGE
%   y=digamma(x);
*/
double digamma1(double x){
double y=0;
  if (x<0) {y=mxGetNaN(); return(y);}
  while ( x < 10 ){
    y -= 1.0 / x;
    x += 1.0;
  }
  y += log(x)-0.5/x;
  x=1.0/(x*x);
  y += ((((((-8.33333333333333333333e-2*x  \
          +2.10927960927960927961e-2)*x \
          -7.57575757575757575758e-3)*x \
          +4.16666666666666666667e-3)*x \
          -3.96825396825396825397e-3)*x \
          +8.33333333333333333333e-3)*x \
          -8.33333333333333333333e-2)*x;
  return(y);
}


/*************************************
 An ANSI-C implementation of the digamma-function for real arguments based
 on the Chebyshev expansion proposed in appendix E of 
 http://arXiv.org/abs/math.CA/0403344 . This is identical to the implementation
 by Jet Wimp, Math. Comp. vol 15 no 74 (1961) pp 174 (see Table 1).
 For other implementations see
 the GSL implementation for Psi(Digamma) in
 http://www.gnu.org/software/gsl/manual/html_node/Psi-_0028Digamma_0029-Function.html

Richard J. Mathar, 2005-11-24
**************************************/

#ifndef M_PIl
/** The constant Pi in high precision */
#define M_PIl 3.1415926535897932384626433832795029L
#endif
#ifndef M_GAMMAl
/** Euler's constant in high precision */
#define M_GAMMAl 0.5772156649015328606065120900824024L
#endif
#ifndef M_LN2l
/** the natural logarithm of 2 in high precision */
#define M_LN2l 0.6931471805599453094172321214581766L
#endif

/** The digamma function in long double precision.
* @param x the real value of the argument
* @return the value of the digamma (psi) function at that point
* @author Richard J. Mathar
* @since 2005-11-24
*/
double digamma2(double x)
{
	/* force into the interval 1..3 */
	if( x < 0.0L )
		return digamma2(1.0L-x)+M_PIl/tanl(M_PIl*(1.0L-x)) ;	/* reflection formula */
	else if( x < 1.0L )
		return digamma2(1.0L+x)-1.0L/x ;
	else if ( x == 1.0L)
		return -M_GAMMAl ;
	else if ( x == 2.0L)
		return 1.0L-M_GAMMAl ;
	else if ( x == 3.0L)
		return 1.5L-M_GAMMAl ;
	else if ( x > 3.0L)
		/* duplication formula */
		return 0.5L*(digamma2(x/2.0L)+digamma2((x+1.0L)/2.0L))+M_LN2l ;
	else
	{
		static long double Kncoe[] = { .30459198558715155634315638246624251L,
		.72037977439182833573548891941219706L, -.12454959243861367729528855995001087L,
		.27769457331927827002810119567456810e-1L, -.67762371439822456447373550186163070e-2L,
		.17238755142247705209823876688592170e-2L, -.44817699064252933515310345718960928e-3L,
		.11793660000155572716272710617753373e-3L, -.31253894280980134452125172274246963e-4L,
		.83173997012173283398932708991137488e-5L, -.22191427643780045431149221890172210e-5L,
		.59302266729329346291029599913617915e-6L, -.15863051191470655433559920279603632e-6L,
		.42459203983193603241777510648681429e-7L, -.11369129616951114238848106591780146e-7L,
		.304502217295931698401459168423403510e-8L, -.81568455080753152802915013641723686e-9L,
		.21852324749975455125936715817306383e-9L, -.58546491441689515680751900276454407e-10L,
		.15686348450871204869813586459513648e-10L, -.42029496273143231373796179302482033e-11L,
		.11261435719264907097227520956710754e-11L, -.30174353636860279765375177200637590e-12L,
		.80850955256389526647406571868193768e-13L, -.21663779809421233144009565199997351e-13L,
		.58047634271339391495076374966835526e-14L, -.15553767189204733561108869588173845e-14L,
		.41676108598040807753707828039353330e-15L, -.11167065064221317094734023242188463e-15L } ;

		register long double Tn_1 = 1.0L ;	/* T_{n-1}(x), started at n=1 */
		register long double Tn = x-2.0L ;	/* T_{n}(x) , started at n=1 */
		register long double resul = Kncoe[0] + Kncoe[1]*Tn ;

		x -= 2.0L ;

		for(int n = 2 ; n < sizeof(Kncoe)/sizeof(long double) ;n++)
		{
			const long double Tn1 = 2.0L * x * Tn - Tn_1 ;	/* Chebyshev recursion, Eq. 22.7.4 Abramowitz-Stegun */
			resul += Kncoe[n]*Tn1 ;
			Tn_1 = Tn ;
			Tn = Tn1 ;
		}
		return resul ;
	}
}

/* call Matlab psi */
double digammam(double x){
  double y;
  mxArray *rhs[1], *lhs[1];
  rhs[0] = mxCreateDoubleMatrix(1, 1, mxREAL);
  *mxGetPr(rhs[0]) = x;
  mexCallMATLAB(1, lhs, 1, rhs, "digamma");
  y=*mxGetPr(lhs[0]);
  /* Clean up allocated memory. */
  mxDestroyArray(rhs[0]);
  mxDestroyArray(lhs[0]);
  return(y);
}

/* call Matlab trigamma */
double trigammam(double x){
  double y;
  mxArray *rhs[1], *lhs[1];
  rhs[0] = mxCreateDoubleMatrix(1, 1, mxREAL);
  *mxGetPr(rhs[0]) = x;
  mexCallMATLAB(1, lhs, 1, rhs, "trigamma");
  y=*mxGetPr(lhs[0]);
  /* Clean up allocated memory. */
  mxDestroyArray(rhs[0]);
  mxDestroyArray(lhs[0]);
  return(y);
}

/* The digamma function is the derivative of gammaln.

   Reference:
    J Bernardo,
    Psi ( Digamma ) Function,
    Algorithm AS 103,
    Applied Statistics,
    Volume 25, Number 3, pages 315-317, 1976.

    From http://www.psc.edu/~burkardt/src/dirichlet/dirichlet.f
    (with modifications for negative numbers and extra precision)
*/
double digamma3(double x)
{
  double neginf = -INFINITY;
  static const double c = 12,
    digamma1 = -0.57721566490153286,
    trigamma1 = 1.6449340668482264365, /* pi^2/6 */
    s = 1e-6,
    s3 = 1./12,
    s4 = 1./120,
    s5 = 1./252,
    s6 = 1./240,
    s7 = 1./132,
    s8 = 691./32760,
    s9 = 1./12,
    s10 = 3617./8160;
  double result;
  /* Illegal arguments */
  if((x == neginf) || isnan(x)) {
    return NAN;
  }
  /* Singularities */
  if((x <= 0) && (floor(x) == x)) {
    return neginf;
  }
  /* Negative values */
  /* Use the reflection formula (Jeffrey 11.1.6):
   * digamma(-x) = digamma(x+1) + pi*cot(pi*x)
   *
   * This is related to the identity
   * digamma(-x) = digamma(x+1) - digamma(z) + digamma(1-z)
   * where z is the fractional part of x
   * For example:
   * digamma(-3.1) = 1/3.1 + 1/2.1 + 1/1.1 + 1/0.1 + digamma(1-0.1)
   *               = digamma(4.1) - digamma(0.1) + digamma(1-0.1)
   * Then we use
   * digamma(1-z) - digamma(z) = pi*cot(pi*z)
   */
  /* Use Taylor series if argument <= S */
  if(x <= s) return digamma1 - 1/x + trigamma1*x;
  /* Reduce to digamma(X + N) where (X + N) >= C */
  result = 0;
  while(x < c) {
    result -= 1/x;
    x++;
  }
  /* Use de Moivre's expansion if argument >= C */
  /* This expansion can be computed in Maple via asympt(Psi(x),x) */
  if(x >= c) {
    double r = 1/x, t;
    result += log(x) - 0.5*r;
    r *= r;
#if 0
    result -= r * (s3 - r * (s4 - r * (s5 - r * (s6 - r * s7))));
#else
    /* this version for lame compilers */
    t = (s5 - r * (s6 - r * s7));
    result -= r * (s3 - r * (s4 - r * t));
#endif
  }
  return result;
}

// TRIGAMMA Computes the trigamma function for positive arguments
// The trigamma function is the second derivative of the loggamma function.
// USAGE
//    y=trigamma(x);
double trigamma1(double x){
double y=0, p;
  while ( x < 12 ){
    y += 1.0 / (x*x);
    x += 1.0;
  }
  p = 1.0/(x*x);
  y += (((((((((1.1666666666666667*p) \
               -0.2531135531135531)*p) \
               +0.7575757575757576e-1)*p  \
               -0.3333333333333333e-1)*p \
               +0.2380952380952381e-1)*p \
               -0.3333333333333333e-1)*p \
               +0.1666666666666667)*p+1.0)/x + 0.5*p;
  return(y);
}

/******************************************************************************/
double trigamma2 ( double x)
/******************************************************************************/
/*
  Purpose:
    TRIGAMMA calculates trigamma(x) = d^2 log(gamma(x)) / dx^2
  Licensing:
    This code is distributed under the GNU LGPL license. 
  Modified:
    08 November 2010
  Author:
    Original FORTRAN77 version by BE Schneider.
    C version by John Burkardt.
  Reference:
    BE Schneider,
    Algorithm AS 121:
    Trigamma Function,
    Applied Statistics, 
    Volume 27, Number 1, pages 97-99, 1978.
  Parameters:
    Input, double X, the argument of the trigamma function.
    0 < X.
    Output, int *IFAULT, error flag.
    0, no error.
    1, X <= 0.
    Output, double TRIGAMMA, the value of the trigamma function at X.
*/
{
  double a = 0.0001;
  double b = 5.0;
  double b2 =  0.1666666667;
  double b4 = -0.03333333333;
  double b6 =  0.02380952381;
  double b8 = -0.03333333333;
  double value;
  double y;
  double z;
/*   Check the input. */
  z = x;
/*   Use small value approximation if X <= A. */
  if ( x <= a ){
    value = 1.0 / x / x;
    return value;
  }
/*   Increase argument to ( X + I ) >= B. */
  value = 0.0;
  while ( z < b )  {
    value = value + 1.0 / z / z;
    z = z + 1.0;
  }
/*   Apply asymptotic formula if argument is B or greater. */
  y = 1.0 / z / z;
  value = value + 0.5 * 
      y + ( 1.0 
    + y * ( b2  
    + y * ( b4  
    + y * ( b6  
    + y *   b8 )))) / z;
  return value;
}
/******************************************************************************/


/* The trigamma function is the derivative of the digamma function.

   Reference:

    B Schneider,
    Trigamma Function,
    Algorithm AS 121,
    Applied Statistics, 
    Volume 27, Number 1, page 97-99, 1978.

    From http://www.psc.edu/~burkardt/src/dirichlet/dirichlet.f
    (with modification for negative arguments and extra precision)
*/
double trigamma3(double x)
{
  double neginf = -INFINITY,
    small = 1e-4,
    large = 8,
    trigamma_1 = 1.6449340668482264365, /* pi^2/6 = Zeta(2) */
    tetragamma_1 = -2.404113806319188570799476,  /* -2 Zeta(3) */
    b2 =  1./6,  /* B_2 */
    b4 = -1./30, /* B_4 */
    b6 =  1./42, /* B_6 */
    b8 = -1./30, /* B_8 */
    b10 = 5./66; /* B_10 */
  double result;
  /* Illegal arguments */
  if((x == neginf) || isnan(x)) {
    return NAN;
  }
  /* Singularities */
  if((x <= 0) && (floor(x) == x)) {
    return neginf;
  }
  /* Negative values */
  /* Use the derivative of the digamma reflection formula:
   * -trigamma(-x) = trigamma(x+1) - (pi*csc(pi*x))^2
   */
  if(x < 0) {
    result = M_PI/sin(-M_PI*x);
    return -trigamma3(1-x) + result*result;
  }
  /* Use Taylor series if argument <= small */
  if(x <= small) {
    return 1/(x*x) + trigamma_1 + tetragamma_1*x;
  }
  result = 0;
  /* Reduce to trigamma(x+n) where ( X + N ) >= B */
  while(x < large) {
    result += 1/(x*x);
    x++;
  }
  /* Apply asymptotic formula when X >= B */
  /* This expansion can be computed in Maple via asympt(Psi(1,x),x) */
  if(x >= large) {
    double r = 1/(x*x), t;
#if 0
    result += 0.5*r + (1 + r*(b2 + r*(b4 + r*(b6 + r*(b8 + r*b10)))))/x;
#else
    t = (b4 + r*(b6 + r*(b8 + r*b10)));
    result += 0.5*r + (1 + r*(b2 + r*t))/x;
#endif
  }
  return result;
}


int nonconv=0;
double maxr=0;

#define max(a,b) (((a)>(b))?(a):(b))

// dirsolve Converts expected sufficient statistics into parameter values
// ess : expected sufficient statistics
// ap contains initial guess of a on entry, the posterior on return
// ess and ap are both q vectors, ws is a workspace memory vector with
// 2q elements
// Uses Newton's method
void dirsolve(double *ess, double *ap, int q, double *ws){
double tol=2e-12, *r, eta, psieta, *psi1a, *da, sump1a, sumrp, lam;
int maxit=50, i, j;
bool check;
  r=ws;
  psi1a=ws+q;
  da = psi1a;
  for (j=0; j<q; ++j) {if (ess[j]<-1.797e308){ {ap[j]=0;} }}
  for (i=1; i<=maxit; ++i) { 
    eta=0; for (j=0; j<q; ++j) {eta += ap[j];}
    //psieta=psi(eta);
    psieta=digamma1(eta);
    check=1;
    for (j=0; j<q; ++j) {
      r[j] = ess[j]<-1e308 ? 0 : digamma1(ap[j]) - psieta - ess[j];  
      if (fabs(r[j]*max(1,ess[j])) > tol) check=0;
    }
    if (check) break;    // convergence check
    // update a
    sump1a=0;  sumrp=0;
    for (j=0; j<q; ++j) {
        psi1a[j] = 1.0/trigamma1(ap[j]); 
        sump1a += psi1a[j];
        sumrp += r[j]*psi1a[j];
    }
    lam = 1;
    sumrp /= ((1.0/trigamma1(eta)) - sump1a);
    for (j=0; j<q; ++j) {
        da[j] = (r[j] + sumrp)*psi1a[j];
        while (ap[j] < lam*da[j]) lam /= 2; 
    }
    for (j=0; j<q; ++j) {
      ap[j] -= lam*da[j];
    }
  }
  if (i>=maxit) {nonconv +=1; for (j=0; j<q; ++j) maxr = maxr>=fabs(r[j]) ? maxr : fabs(r[j]);}
}


void mexFunction(
   int nlhs, mxArray *plhs[],
   int nrhs, const mxArray *prhs[])
{
  size_t n, j, k;
  int q, i;
  double *a, *ess, *ws, *ap;
  
  /* Error checking on inputs */  
  if (nrhs<2 || nrhs>2) mexErrMsgTxt("Wrong # of input arguments.");
  if (nlhs>1) mexErrMsgTxt("Too many output arguments.");
  for (i=0; i<2; ++i){
    if (!mxIsDouble(prhs[i]) && !mxIsSparse(prhs[i]))
        mexErrMsgTxt("Inputs must be double and full");
    if (mxIsComplex(prhs[i]))
        mexErrMsgTxt("Inputs must be real.");
  }
   
  // ess and a are both q x n
  q=(int)mxGetM(prhs[0]);
  n=mxGetN(prhs[0]);
  if (mxGetM(prhs[1])!=q || mxGetN(prhs[1])!=n)
      mexErrMsgTxt("ess and a must be the same size");
  
  ess=mxGetPr(prhs[0]); 
  plhs[0]=mxDuplicateArray(prhs[1]); // copy initial a into output array
  ap=mxGetPr(plhs[0]);
  ws=mxMalloc(2*q*sizeof(double));   // temporary workspace memory
  nonconv=0;
  for (j=0; j<n; ++j) { 
      dirsolve(ess, ap, q, ws);
      ess += q;                
      ap  += q;
  } 
  if (nonconv>0) printf("# of convergence failures in mex: %1.0i  %1.4e\n",nonconv,maxr);
  mxFree(ws);
}


















