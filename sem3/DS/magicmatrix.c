// WAP to read a set of integers for a square matrix,
// then decide whether the matrix represents a magic square or not.

#include<stdio.h>

void main()
{
    int n,a[n][n],i,j,sum1=0,sum2=0;
    printf("\nEnter the Size of Square Matrix : ");
    scanf("%d",&n);
    for(i=0;i<n;i++)
    {   printf("Enter the elements of raw : %d \n",i+1);
        for(j=0;j<n;j++)
        {
            scanf("%d",&a[i][j]);
        }
    }
    for(i=0;i<n;i++)
    {
        sum1=sum1+a[i][i]; // checking 1st diagonal sum
        sum2=sum2+a[i][n-1-i]; //2nd diagonal
    }
    int r[n],c[n]; //to take sum of each raw and each column 
    for(i=0;i<n;i++)
    {   
        int sum=0,jum=0; //sum for raw, jum for column XD
        for(j=0;j<n;j++)
        {   
            sum=sum+a[i][j]; //i=raw ,j=column 
            jum=jum+a[j][i];
        }
        r[i]=sum; //feeding sum of each raw
        c[i]=jum;
    }
    int count=0; 
    if(sum1==sum2)
    {   
        for(i=n-1;i>0;i--)
        {
            if(r[i]==r[i-1] && c[i]==c[i-1])
            {
                count=count+1;
            }
        }
        if(count==n-1)
        {
            printf("Your Matrix is a Magic Square :) \n");
        }
        else
        {
            printf("It is not a Magic Square Matrixx \n");
        }
    }
    else
    {
        printf("It is not a Magic Square :_: \n");
    }
}