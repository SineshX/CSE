//Q1: WAP to insert any element at any given position
#include<stdio.h>
#include<stdlib.h>
#include<time.h>

void main()
{
    int n=10,a[n],k,item,lower=0,upper=100;
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
    printf("\nEnter position where you wanna inset an element : ");
    scanf("%d",&k);
    printf("Enter the elememt that you wanna insert : ");
    scanf("%d",&item);

    k=k-1;
    n=n+1;
    int i;
    for(i=n;i>k;i--)
    {
        a[i]=a[i-1];
    }
    a[i]=item;

    printf("\nNew Array ...\n");
    for(int i=0;i<n;i++){ 
        printf("\n%d.) %d",i+1,a[i]);
    }
    printf("\n");
}