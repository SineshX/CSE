#include<stdio.h>
#include<stdlib.h>
int a;
typedef struct node
{
    int data;
    struct node *next;
}node;

void printList(node *head);
void insertBeg(node *head);
void insertEnd(node *head);
void insertMid(node **head);

void main()
{   
    system("cls");
    char chr='y';
// create a new node 
    node *head,*ptr,*ptr2;
    printf("Enter the data of first node(head) : ");
    scanf("%d",&a);
    ptr = (node*)malloc(sizeof(node));
    ptr->data= a;
    ptr->next=NULL;
    head = ptr;
    printList(head);
// done

    while (chr=='y')
    {   
        printf("\nWhere do wanna insert the node ?");
        printf("\nTo insert at the Begening   press 1,");
        printf("\nTo insert at the end        press 2,");
        printf("\nTo insert at given position press 3,\n");
        scanf("%d",&a);
        switch(a)
        {
            case 1:
            {
                insertBeg(head);
                break;
            }
            case 2:
            {
                insertEnd(head);
                break;
            }
            case 3:
            {
                insertMid(head);
                break;
            }
            default:
            {
                printf("Please Enter Correct number :) ");
                continue;
            }
        }

        printf("\nDo you wanna make changes in the loop");
        printf("\nPress y for YES and n for NO : ");
        scanf("%s", &chr ); 
        printf("\n%c",chr);
        
        break;
    }
    // ptr2 = (node*)malloc(sizeof(node));
    // ptr2->data= 30;
    // ptr2->next=NULL;
    // ptr->next=ptr2;

}

void printList(node *head)
{   
    printf("\nhere is your linkedlist : ");
    while(head != NULL)
        {   
            printf(" %d\t",head->data);
            head = head->next ;
        }
}

void insertBeg(node *head)
{
    node *ptrbeg;
    printf("Enter the data of node : ");
    scanf("%d",&a);
    ptrbeg = (node*)malloc(sizeof(node));
    ptrbeg->data= a;
    ptrbeg->next=head;
    head = ptrbeg;
    printList(head);
}

void insertEnd(node *head)
{
    node *ptr2,*ptr;
    printf("Enter the data of node : ");
    scanf("%d",&a);
    ptr = head;
    while(ptr->next !=NULL)
    {
      ptr = ptr->next;
    }
    ptr2 = (node*)malloc(sizeof(node));
    ptr2->data = a;
    ptr2->next = NULL;
    ptr->next = ptr2;
    printList(head);
}

void insertMid(node *head)
{   
    int p;
    node *ptr,*ptr2;
    printList(head);
    printf("\nEnter the position where you wanna insert the node : ");
    scanf("%d",&p);
    ptr = head;
    if (p==1)
        insertBeg(head);
    else
    {
        for(int i=1;i<a-1;i++) 
        {
            ptr = ptr->next;  
        } //ptr has address of one previous node ,ref: position given
        //printf("\nEnter the data of node : ");
        //scanf("%d",&a);
        ptr2 = (node*)malloc(sizeof(node));
        ptr2->data = 100;
        ptr2->next = ptr->next;
        ptr->next = ptr->next;
        printList(head);
    }
    
}
