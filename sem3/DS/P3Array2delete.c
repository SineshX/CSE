//Q2: WAP to delete an elememt at any given position.
#include<stdio.h>
#include<stdlib.h>
#include<time.h>

void main()
{
    int n=10,a[n],k,lower=0,upper=100;
    // can adjust n , lower and upper accordingly
    printf("Here is an Array\n");
    // taking input using random function
    srand(time(0)); 
    for(int i=0;i<n;i++)
    {
        int num = (rand() % (upper-lower +1)) + lower;
        a[i]=num; 
        printf("\n%d.) %d",i+1,num);
    }
    printf("\nEnter position from where you wanna delete an elememt : ");
    scanf("%d",&k);
    k=k-1;
    for(int i=k;i>=k;i++)
    {   
        a[i]=a[i+1];
    }
    n=n-1;
    printf("\nNew Array ...\n");
    for(int i=0;i<n;i++){ 
        printf("\n%d.) %d",i+1,a[i]);
    }
    printf("\n");
}