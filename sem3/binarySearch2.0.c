//Q4: WAP to search any given elememt using Binary search ,
// if not found then insert it.
#include<stdio.h>
#include<stdlib.h>
#include<time.h>

int n=10; 

int mybubble(int a[]) //Bubble Sort
{
    int c = 0; //for swap
    //sorting using bubble sort :)
    for (int s = 0; s < n - 1; s++)
    {
        for (int i = 0; i < n-(s+1); i++) // or n-1 simply
        {
            if (a[i] > a[i + 1])
            {   
                c = a[i];
                a[i] = a[i + 1];
                a[i + 1] = c;
            }
        }
    }
    for (int i = 0; i < n; i++)
    {
        printf("\n %d",a[i]);
    }
    return 0;
}

int main()
{  
    int a[n],lower=0,upper=100,item;
    srand(time(0)); 
    for(int i=0;i<n;i++)
    {
        int num = (rand() % (upper-lower +1)) + lower;
        a[i]=num; 
    }
    printf("\nSorted numbers are \n"); 
    mybubble(a);                //BUBBLE SORT KA JAADU
    printf("\nEnter the elememt that you wanna search : ");
    scanf("%d",&item);
    //BINARY SEARCH // half half kar k
    int lb=0,ub=n-1,mid;
    for(int i=0;i<n;i++)
    {  
        mid=(lb+ub)/2;
        if(item==a[mid])
        {
            printf("\nElememt found :)\n\n");
            return 0;
        }
        else if(item>a[mid])
        {
            lb=mid+1;
        }
        else if(item<a[mid])
        {
            ub=mid-1;
        }
    }
    printf("\nElement Not Found in prevoius array:_: \n");
    //Since elememy is not found ...
    n++; //to increase size of array
    a[n-1]=item; //coz, last wale me kucch hai jaata XD
    printf("\nHere is Array with your element\n");
    mybubble(a);
    printf("\n");
    return 0;
}
/*
//Q4: WAP to search any given elememt using Binary search ,
// if not found then insert it.
#include<stdio.h>
const int n=10;
int main()
{
    int a[n];
    int item,num=0;
    printf("Here is an Array\n");
    for(int i=0;i<n;i++)
    {   num=num+10;
        a[i]=num; 
        printf("\n%d.) %d",i+1,a[i]);
    }
    printf("\nEnter the elememt that you wanna search : ");
    scanf("%d",&item);

    int lb=0,ub=n-1,mid;
    for(int i=0;i<n;i++)
    {   mid=(lb+ub)/2;
        
        if(item==a[mid])
        {
            printf("\nElememt found :)\n\n");
            return 0;
        }
        else if(item>a[mid])
        {
            lb=mid+1;
        }
        else if(item<a[mid])
        {
            ub=mid-1;
        }
    }
    printf("\nSorry ;_; Element Not Found in prevoius array:_: \n");
    //Since elememy is not found ...
     //n=n+1;
    a[n]=item; //is it legal to increase size like this ?
    //printf("\n%d%d",n,a[n]);
    printf("\nHere is new Array with your element\n");
    for(int i=0;i<=n;i++)
     {  
         printf("\n%d.) %d",i+1,a[i]);
     }
     printf("\n"); 
     return 0;
}
*/
