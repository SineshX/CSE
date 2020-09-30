#include<stdio.h>
#include<stdlib.h> 
#include<string.h> 

int main(void)
{   
    int n,value;
    char exp[100],stack[100];
    printf("please Enter postfix expression ");
    scanf("%s",&exp);
    n = strlen(exp);
    for(int i=0;i<n;i++)
    {
        switch(exp[i])
        {
            case '0':
                push(0);
                break;
            case '1':
                push(1);
                break;
            case '2':
                push(2);
                break;
            case '3':
                push(3);
                break;
            case '4':
                push(4);
                break;
            case '5':
                push(5);
                break;
            case '6':
                push(6);
                break;
            case '7':
                push(7);
                break;
            case '8':
                push(8);
                break;
            case '9':
                push(9);
                break;
            
        }
    }

    return 0;
}