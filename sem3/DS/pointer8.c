//WAP to max of two no. using pointers.

#include<stdio.h>

int mypointermax(int *x,int *y)
{   
    if (*x > *y)
        printf("Maximum of two numbers(%d,%d)is : %d\n",*x,*y,*x);
    else if (*y > *x)
        printf("Maximum of two numbers(%d,%d)is : %d\n",*x,*y,*y);
    else
        printf("both numbers(%d,%d) are equal\n",*x,*y);
}
void main()
{   
    int a,b;
    printf("Please Enter Two numbers\n");
    scanf("%d%d",&a,&b);
    mypointermax(&a,&b);
}