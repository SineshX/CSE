/*  
Q1:  Write a short C program that declares and initializes (to any value you like) a
double, an int, and a char. Next declare and initialize a pointer to each of
the three variables. Your program should then print the address of, and value
stored in, and the memory size (in bytes) of each of the six variables. 

*/
#include<stdio.h>

int main(void)
{
    int a = 10 ;
    char b = 'A';
    float c= 30.59;
    double d = 96.375423;
    int *p1;
    char *p2;
    float *p3;
    double *p4;
    p1 = &a;
    p2 = &b;
    p3 = &c;
    p4 = &d;
    printf(" a(int*) = %d and address of a is\t 0x%p,\n",*p1,p1);
    printf(" b(char*) = %c and address of b is\t 0x%p,\n",*p2,p2);
    printf(" c(float*) = %.5f and address of c is 0x%p,\n",*p3,p3);
    printf(" d(double*) = %.4f and address of d is 0x%p,\n",*p4,p4);
    printf("\n");
    printf(" a(int) = %d and address of a is\t 0x%p,\n",a,&a);
    printf(" b(char) = %c and address of b is\t 0x%p,\n",b,&b);
    printf(" c(float) = %.5f and address of c is 0x%p,\n",c,&c);
    printf(" d(double) = %.4f and address of d is 0x%p,\n",d,&d);
    printf("\n");
    printf(" Size of a(int) = %d bytes & size of pointers *p1(int) = %d bytes,\n",sizeof(a),sizeof(*p1) );
    printf(" Size of b(char) = %d bytes & size of pointers *p2(char) = %d bytes,\n",sizeof(b),sizeof(*p2) );
    printf(" Size of c(float)= %d bytes & size of pointers *p3(float) = %d bytes,\n",sizeof(c),sizeof(*p3) );
    printf(" Size of d(double)= %d bytes & size of pointers *p4(double) = %d bytes,\n",sizeof(d),sizeof(*p4) );
    
    //%d will give in int , %p or %x will give address in hexadecimal. 
    
    return 0;
}