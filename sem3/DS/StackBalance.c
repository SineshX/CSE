#include<stdio.h>
#include<stdlib.h>
#include<string.h>

int myStack(char arr[],int n);
void myPop(); //remove top
void myPush(char a); //insert
typedef struct node
{
    char data;
    struct node *next;

}box;
box *top,*tp;

int main()
{   
    char arr[100];
    printf("Please your equation : ");
    scanf("%s",&arr);
    //printf("%s\n",arr);
    int n = strlen(arr); //to get length
    //printf("%d\n",n);
    myStack(arr,n);
    if(top == NULL)
    {
        printf("\nYour Equation is Balanced\n");
    }
    else
    {
        printf("\nYour Equation is  Not Balanced\n");
    }
    return 0;
}

int myStack(char arr[],int n)
{   
    top = NULL; // first time push karne me kaam aayega jab tp = top karenge
    for(int i=0;i<n;i++) // jitne length ka equation hai utne time chalega
    {   
        if( arr[i]=='(' || arr[i]=='{' || arr[i]=='[' ) 
        {   
            myPush(arr[i]);  // top me insert
        }
        else if( arr[i]==')' || arr[i]=='}' || arr[i]==']'  )
        {   
            if(top == NULL)
            { //underflow k case me
                printf("\nYour Equation is Not Balanced\n");
                return 0;
            }
            else
            {
               myPop();  //remove top elememt 
            }
        }
    }
    return 0;
}

void myPop()
{ 
    tp = top;
    top = tp->next;
    //printf("you poped %c\n",tp->data);
    free(tp);
}

void myPush(char a)
{   
    tp = top;
    top = (box*)malloc(sizeof(box));
    top->data= a;
    top->next=tp;
    //printf("you pushed %c\n",a);
}