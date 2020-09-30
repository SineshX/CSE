#include<stdio.h>

void main()
{   int a[10],sum=0;
    //float avg=0;  
    printf("Please enter 10 integers \n");
    for(int i=0;i<10;i++)
    {
        scanf("%d",&a[i]);
    }
   
    for(int i=0;i<10;i++)
    {
       sum=sum+a[i];
    }
    printf("Sum of the numbers is : %d",sum);
    printf("\nAverage of the numbers : %.2f\n",sum/10.0);
    
}