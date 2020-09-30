#include<stdio.h>
#include<stdlib.h>
#include<string.h>

int top=-1,tp=-1,n;
char stack[10],postfix_exp[100]; //my stack and post fix exp : P

/*---------------------------------------------------*/
void push(char arr[],char a,int *top);  // array ka v ref do bete ::
void pop(char arr[],int *top);
int precidence(char arr[],char symb,int *top);
/*---------------------------------------------------*/
int main(void)
{
    char exp[100]; //infix_exp
    printf("Please Enter the expression : ");
    scanf("%s",&exp);
    printf("%s",exp);
    n  = strlen(exp);
    for(int i=0;i<n;i++)
    {
        if(exp[i]== '[' || exp[i]== '{' || exp[i]== '(' )
        {
            push(stack,exp[i],&top);
        }
        if(exp[i]== ']' || exp[i]== '}' || exp[i]== ')' )
        {   
            while(stack[top] != ']' || stack[top] != '}' || stack[top]== ')' )
            {
                if(stack[top]== '^' || stack[top]== '/' || stack[top]== '*' || stack[top]== '+' || stack[top]== '-')
                    {
                        push(postfix_exp,stack[top],&tp);
                    }
                pop(stack,&top);
            }
            pop(stack,&top);
        }
        if(exp[i]== '^' || exp[i]== '/' || exp[i]== '*' || exp[i]== '+' || exp[i]== '-')
        {   
            if( precidence(stack,exp[i],&top) ) //if exp[i] has low precidence than stack[top]
            {   // 3 ways XD
                //1)
                char temp = stack[top]; // temp has high precidence 
                stack[top] = exp[i]; // overwrite xd ,low wala ander gaya 
                push(stack,temp,&top);

                continue;
            }
            push(stack,exp[i],&top);// if high than be on the top
        } 
        else //a to z or number // will go to postfix exp
        {
            push(postfix_exp,exp[i],&tp);
        }
    }

    printf("\nhere is your postFix Expression : ");
    printf("%s",postfix_exp);
    // for(int i=0;i<=tp;i++)
    // {
    //     printf("%d",&postfix_exp[i]);
    // }
    return 0;
}
/*---------------------------------------------------*/
void push(char arr[],char a,int *top)
{
    if(*top + 1 == n){
        printf("\nStack OverFlow :)\n");
        return;
    }
    *top = *top + 1 ;
    arr[*top]=a;
}
/*---------------------------------------------------*/
void pop(char arr[],int *top)
{
    if(*top == -1){
        printf("\nStack UnderFlow :_:\n");
        return;
    }
    printf("\nYou poped out %d\n",arr[*top]);
    *top = *top -1 ; // why not *top--;
}
/*---------------------------------------------------*/
int precidence(char arr[],char symb,int *top)
{ //low precedence krna hai 
    if((symb == '/' || symb == '*' || symb == '+' || symb == '-') && (stack[*top]== '^') )
    {
        return 1; // matlb aanewale ka precidence low hai ,
    }
    if((symb == '+' || symb == '-') && (stack[*top] == '/' || stack[*top] == '*' ) )
    {
        return 1; // matlb aanewale ka precidence low hai ,
    }
    else //^ always  // baki k condition s 
    {
        return 0;
    }
}