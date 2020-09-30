#include<stdio.h>
#include<stdlib.h>
#include<time.h>

int myrandom();
void mystack();
void printStack();
void manualStack();
void myPop();
void myPush();
typedef struct node
{
    int data;
    struct node *next;

}box;
box *top,*tp;
const int n;
int count;

int main(void)
{   
    printf("How many stack box do you wanna create : ");
    scanf("%d",&n);
    srand(time(0));
    //manualStack(n);
    mystack(n);
    printStack();
    int a;
    while(1)
    {
        printf("\nTo Pop  item (delete top) press 1,");
        printf("\nTo Push item (insert top) press 2,");
        printf("\nTo Exit press 0,\n");
        scanf("%d",&a);
        if(a==0)
        {
            break;
        } 
        switch(a)
        {
            case 1:
            {   
                if(top==NULL)
                {
                    printf("\nStack Underflow :_: Empty Stack\n\n");
                    break;
                }
                myPop();
                break;
            }
            case 2:
            {   
                if(count==n)
                {
                    printf("\nStack Overflow ;_; Limit reached\n\n");
                    break;
                }
                myPush();
                break;
            }
        }
        printStack();
    } //end of while loop
    return 0;
}

int myrandom()
{   int lower=0,upper=100,num;
    // srand(time(0));
    num = (rand() % (upper-lower +1)) + lower;
    return num;
}

void mystack()
{
    tp = NULL;
    for(int i=0;i<n;i++)
    {
        top = (box*)malloc(sizeof(box));
        top->data= myrandom();
        top->next=tp;
        tp = top;
    }
}

void printStack()
{   
    count = 0;
    printf("Elememts of the Stack are : ");
    tp = top;
    while(tp!=NULL)
    {   
        count = count+1;
        printf("%d\t",tp->data);
        tp = tp->next;
    }

}

void manualStack()
{   int a;
    tp = NULL;
    for(int i=0;i<n;i++)
    {   
        printf("Please Enter The data of stack %d : ",i+1);
        scanf("%d",&a);
        top = (box*)malloc(sizeof(box));
        top->data= a;
        top->next=tp;
        tp = top;
    } 
}

void myPop()
{
    tp = top;
    top = tp->next;
    printf("you poped %d\n",tp->data);
    free(tp);
}

void myPush()
{
    int a;
    tp = top;  
    printf("Enter The data of top stack : ");
    scanf("%d",&a);
    top = (box*)malloc(sizeof(box));
    top->data= a;
    top->next=tp;
}