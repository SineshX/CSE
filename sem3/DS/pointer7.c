//WAP to add two numbers using call by reference

#include<stdio.h>

int mypointersum(int *x,int *y)
{
   return *x+*y;
}
void main()
{   
    int a,b,sum;
    printf("Please Enter Two numbers\n");
    scanf("%d%d",&a,&b);
    sum = mypointersum(&a,&b);
    printf("Sum of two numbers is : %d\n",sum );
}