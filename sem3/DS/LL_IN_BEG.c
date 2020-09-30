#include<stdio.h>
#include<stdlib.h>


typedef struct node
{
    int data;
    struct node *next;
}node;

int main()
{
    int a,i;
    int ch=1;
    node *head,*ptr,*ptr2;
    ptr = (node*)malloc(sizeof(node));
    printf("Enter The data of node (head): ");
    scanf("%d",&a);
    ptr->data = a;
    ptr->next = NULL;
    head = ptr;
    printf("Do You wanna Insert data before head Node ? \npress 1 for YES and 0 for NO : ");
    scanf("%d",&ch);
    while(ch==1)
    {   ptr = (node*)malloc(sizeof(node));
        printf("Enter The data of node : ");
        scanf("%d",&a);
        ptr->data = a; 
        ptr->next = head;
        head = ptr;
        printf("Do You wanna Insert data Before Head node? \npress 1 for YES and 0 for NO : ");
        scanf("%d",&ch);
    }

    while(ptr != NULL)
    {
        printf("%d\t",ptr->data);
        ptr=ptr->next;
    }
    
    return 0;
}

