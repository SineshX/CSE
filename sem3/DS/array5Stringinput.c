#include<stdio.h>
#include<string.h>

void main()
{
    char a[100];
    printf("\n\nPlease enter string \n\n");
    gets(a);
    int length = strlen(a);
    printf("\nlength of your string is %d\n\n",length);

}