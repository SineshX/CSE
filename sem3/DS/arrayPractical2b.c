#include<stdio.h>
#include<stdlib.h>
#include<time.h>

void main()
{   printf("\n");
    int a[25],lower=0,upper=25;
    srand(time(0)); //what is this :_:
    for(int i=0;i<25;i++)
    {
        int num = (rand() % (upper-lower +1)) + lower;
        a[i] = num;
    }
    for(int i=0;i<25;i++)
    {
        printf("%d\n",a[i]);
    }
    //total pairs n(n+1)/2 // here n=25-1 // just for info XD
    //a: Find all pairs whose sum is 25
    int count=0;
    for(int i=0;i<24;i++){
       for(int j=1;j<25;j++){
            if(a[i]+a[i+1] == 25){
                count = count + 1 ;
            }
        }
    }
    printf("\nThere are %d pairs whose Sum is 25 \n",count);
    
    //b: count even and odd numbers
    int even=0,odd=0;
    for(int i=0;i<25;i++)
    {
        if(a[i]%2==0){
            even=even+1;
        }
        else{
            odd=odd+1;
        }
    } 
 printf("\nEVNUM ( Even Numbers ) = %d \nODNUM ( Odd Numbers ) = %d\n\n",even,odd);
}