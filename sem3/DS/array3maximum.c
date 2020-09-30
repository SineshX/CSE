#include<stdio.h>

void main()
{   
    int a[10];  
    printf("Please enter 10 integers \n");
    for(int i=0;i<10;i++)
    {
        scanf("%d",&a[i]);
    }
    int max = a[0];
    for(int i=1;i<10;i++)
    {
        if(max<a[i])
        {
            max = a[i];
        }
    }
    printf("%d is the largest number\n",max);
}