// Practical 1 : Q1 : Solution 
#include<stdio.h>

void main()
{   
    int a,b,c;
    printf("Please Enter 3 numbers \n");
    scanf("%d%d%d",&a,&b,&c);
    if (a>b)
    {   
        if(a>c)
        printf("%d is the largest number among %d, %d, %d\n",a,a,b,c);
        else 
        printf("%d is the largest number among %d, %d, %d\n",c,a,b,c);
    }
    else
    {
        printf("%d is the largest number among %d, %d, %d\n",b,a,b,c);
    }
}