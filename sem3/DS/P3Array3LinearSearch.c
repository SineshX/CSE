//Q3: WAP to search any given elememt using linear search.
//one by one XD
#include<stdio.h>
#include<stdlib.h>
#include<time.h>

int main()
{
    int n=10,a[n],item,lower=0,upper=100;
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
    printf("\nEnter the elememt that you wanna search : ");
    scanf("%d",&item);
   
    for(int i=0;i<n;i++)
    {
        if(item==a[i]){
            printf("\nElememt %d found at position : %d \n",item,i+1);
            return 0;
        }
    }
    printf("\nSorry ;_; Element Not Found :_: \n");
    return 0;
}