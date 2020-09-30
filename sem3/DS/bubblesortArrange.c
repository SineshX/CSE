//Q2: WAP to arrange numbers in ascending order using Bubble Sort.
#include <stdio.h>
#include<stdlib.h>
#include<time.h>
const int n=8;

int main()
{
    int a[n],lower=0,upper=100;
    printf("\nHere is unsorted numbners\n");
    srand(time(0)); 
    for(int i=0;i<n;i++)
    {
        int num = (rand() % (upper-lower +1)) + lower;
        a[i]=num; 
        printf("\n %d",num);
    }
    printf("\n");
    int c = 0; //for swap
    //sorting using bubble sort :)
    for (int s = 0; s < n - 1; s++)
    {
        for (int i = 0; i < n-(s+1); i++)  //
        {
            if (a[i] > a[i + 1])
            {   
                c = a[i];
                a[i] = a[i + 1];
                a[i + 1] = c;
            }
        }
    }
    printf("\nNumbers in Ascending order :  \n");
    for (int i = 0; i < n; i++)
    {
        printf("\n %d",a[i]);
    }
    printf("\n");
    return 0;
}
