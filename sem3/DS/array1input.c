#include<stdio.h>

void main()
{   int a[10];  
    printf("Please enter 10 integers \n");
    for(int i=0;i<10;i++)
    {
        scanf("%d",&a[i]);
    }
    printf("You have entered :\n"); //gcc -g test.c -o ./test
    for(int i=0;i<10;i++)
    {
       printf("%d\t",a[i]);
    }
    printf("\n");
}