#include<stdio.h>

int main()
{   
    int i,j,count=0;
    for(i=0,j=0;j<20,i<10;i++,j++)
    {
        printf("i=%d, j=%d\n",i,j);
        count++;
    }
    printf("%d %d %d",count,i,j);
}