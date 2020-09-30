//Infix to postfix // try 2
#include<stdio.h>
#include<stdlib.h>
#include<string.h>

char stack[100],postfix[100];
 int top = -1 , tp = -1 ;


void push_into_stack(char a);
void push_into_postfix(char a);
int precedence(char a);
void pop_from_stack_and_push_into_postfix();

void main()
{
//    char exp[100];
  /*  printf("Please your infix expressio : ");
    scanf("%s",&exp);
    int n = strlen(exp); //to get length
    //printf("%d\n",n);
    exp[n] = ')';
    printf("%s\n",exp);
    push_into_stack('(');
    */
    char exp[] = "a+b-c/d";
    int n = strlen(exp);
    for(int i=0;i<n;i++)
    {
        if(exp[i] == '(')
        {
            push_into_stack(exp[i]);
        }
        else if (exp[i] == '+' || exp[i] == '-' || exp[i] == '/'  || exp[i] == '*' || exp[i] == '^' ) //operator
        {

            printf("checking precedence :)\n");
            if( precedence(exp[i])>precedence(stack[top]) )  // compare exp[i] with stack[top]
            {
                push_into_stack(exp[i]);
            }
            else // low precedence ,means we need to exchnge
            {
                push_into_postfix(stack[top]);
                stack[top] = exp[i]; // pushed or u can say replaced
            }

        }
        else if (exp[i] == ')') //start popping
        {
            while(stack[top] != '(')
            {
                pop_from_stack_and_push_into_postfix(); // stack k top ko
            }
            //pop_stack();
            top = top - 1;
        }
        else //alphabet or number
        {
            push_into_postfix(exp[i]);
        }

    }
    while(stack[top] != '(')
    {
    pop_from_stack_and_push_into_postfix(); // stack k top ko
    }
    printf("\nyour postfix expression is : %s\n",postfix);
}
/*--------------------------------------------------*/
void push_into_stack(char a)
{
    top++;
    stack[top]=a;
}

void push_into_postfix(char a)
{
    tp++;
    postfix[tp]=a;
}

void pop_from_stack_and_push_into_postfix()
{
    push_into_postfix(stack[top]);
    top = top - 1;
}

int precedence(char a)
{
    switch (a)
    {
    case '+':
    case '-':
        return 1;

    case '*':
    case '/':
        return 2;

    case '^':
        return 3;
    }
    return -1;
}