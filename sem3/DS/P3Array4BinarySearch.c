//Q4: WAP to search any given elememt using Binary search.
//half half XD
#include<stdio.h>


int main()
{
    int n=10,a[n],item,num=0;
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
        //printf("%d,%d,%d\n",lb,ub,mid);
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
    //Since elememy is not found :_:
    n=n+1;
    a[n+1]=item;

    //printf("\nSorry ;_; Element Not Found :_: \n\n");
    return 0;
}