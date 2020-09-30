#include<stdio.h>

void main(){
    int n,rem,rev=0;
    printf("\nEnter the five digit number : ");
    scanf("%d",&n);
    
    for(int i=0;i<5;i++)  
    {
        rem = n % 10;
        rev = rev*10+rem;
        n = n/10;
    }

    printf("\nYour number is reversed : %d\n",rev);

}