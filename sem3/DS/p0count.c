#include<stdio.h>

void main()
{   int a[10];
    printf("\nPlease enter 10 positive numbers \n");
    for(int i=0;i<10;i++)
    {
        scanf("%d",&a[i]);
    }
    int even = 0;
    int odd = 0;
    for(int i=0;i<10;i++)
    {
        if(a[i] % 2==0)
        {
            even=even+1;
        }
        else
        {
            odd=odd+1;
        }    
    }

    printf("there are %d even and %d odd numbers \n",even,odd);
}