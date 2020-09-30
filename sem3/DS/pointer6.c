//Q1: WAP to add two numbers using pointers 

#include<stdio.h>

void main()
{   
    int a,b;
    int *x,*y;
    printf("Please Enter Two numbers\n");
    scanf("%d%d",&a,&b);
    x = &a;
    y = &b;
    printf("Sum of two numbers is : %d\n", *x + *y );
}