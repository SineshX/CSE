//searching an element in array
#include<stdio.h>

int main()
{   int size,s;  
    printf("\nPlease enter size of array : ");
    scanf("%d",&size);
    int a[size];

    printf("Please enter integers \n");
    for(int i=0;i<size;i++)
    {
        scanf("%d",&a[i]);
    }
    printf("Which number you wanna search ?");
    scanf("%d",&s);
    for(int i=0;i<size;i++)
    {
       if(s==a[i])
       {  
          printf(":) Number Found \n");
          printf("position of your number is %dth\n",i+1);
          return 0;
       } 
    }
    printf(" :_: Numnber not found\n");
    return 0;
}