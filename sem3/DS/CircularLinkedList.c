// Q2: Circular Linked List
// Write code for following operations on circular linked list,
// #  a Create Linked List
// 1. Insert a node at the beginning of Linked list
// 2. Insert a node at the end of Linked List
// 3. Insert a node in the middle of the linked list
// 4. Delete a node at the beginning  of linked list
// 5. Delete a node at the end of linked list
// 6. Delete a node in the middle of linked list
// Write menu driven program for circular linked list for above said operations.
// All above said operations need to be implemented with the help of functions.

#include<stdio.h>
#include<stdlib.h>
#include<time.h>
int a,n;  
int myrandom();
struct node printList();
struct node insertBeg();
struct node insertEnd();
struct node insertMid();
struct node deleteBeg();
struct node deleteEnd();
struct node deleteMid();
struct node myCircularLL(int  length);

typedef struct node
{
    int data;
    struct node *next;
}box;
box *head,*ptr,*ptr2; 

int main()
{   
    printf("Automatic data will be feeded for initial nodes\n");
    printf("How many node do you wanna create ? : ");
    scanf("%d",&n);
    srand(time(0));
    myCircularLL(n);
    printList();
    //menu driven program for Citcular linked list :)
        while (1)
        {   
            //printf("\nPlease Enter Correct option  :) ");
            printf("\nTo insert at the beginning  press 1,");
            printf("\nTo insert at the end        press 2,");
            printf("\nTo insert at given position press 3,");
            printf("\nTo delete at the beginning  press 4,");
            printf("\nTo delete at the end        press 5,");
            printf("\nTo delete at given position press 6,");
            printf("\nTo exit press 0 :)\n");
            scanf("%d",&a);
            if(a==0){
                break;
            }
            switch(a)
            {
                case 1:
                {
                    insertBeg();
                    break;
                }
                case 2:
                {
                    insertEnd();
                    break;
                }
                case 3:
                {
                    insertMid();
                    break;
                }
                case 4:
                {
                    deleteBeg();
                    break;
                }
                case 5:
                {
                    deleteEnd();
                    break;
                }
                case 6:
                {
                    deleteMid();
                    break;
                }
                // default:
                // {
                //     continue;
                // }
            }
            printList();
        }//end of while loop
    //completed
    printList();
    return 0;
}// end of main function

struct node myCircularLL(int  length)
{   
    head = (box*)malloc(sizeof(box)); //head = 1100000ff
    head->data = myrandom();
    head->next=NULL;
    ptr = head;
    for(int i=1;i<length;i++)
    {
        ptr2 = (box*)malloc(sizeof(box));
        ptr2->data = myrandom();
        ptr2->next = NULL;
        ptr->next = ptr2;
        ptr = ptr->next;
    }
    ptr->next = head;
    //return *head;
} // end of myccLL 

int myrandom()
{   int lower=0,upper=100,num;
    // srand(time(0));
    num = (rand() % (upper-lower +1)) + lower; //33
    return num;
}

struct node printList() //done
{   
    ptr = head;
    printf("Here is your Circular Linked List :");
    do
    {   
        printf(" %d\t",ptr->data);
        ptr = ptr->next ;
    }while(ptr != head);
    //return *head;
}

struct node insertBeg() //1 done
{   
    printf("Enter the data head of node : ");
    scanf("%d",&a);
    ptr2 = (box*)malloc(sizeof(box));
    ptr2->data= a;
    ptr2->next=head;
    ptr=head;
    while(ptr->next!=head){
        ptr=ptr->next; 
    } // now ptr will have location of last node
    ptr->next = ptr2; //feed address of newly created head(ptr2)
    head = ptr2; //now ptr2 is head node
    //return *head;
}

struct node insertEnd() //2 done
{
    printf("Enter the data last of node : ");
    scanf("%d",&a);
    ptr = head;
    while(ptr->next !=head)
    {
      ptr = ptr->next;
    } // now ptr will have location of last node
    ptr2 = (box*)malloc(sizeof(box));
    ptr2->data = a;
    ptr2->next = head; //last node have the add of head
    ptr->next = ptr2;
    //return *head;
}

struct node insertMid() //3 already done
{   int p;
    printf("Enter the position of new node: ");
    scanf("%d",&p);
    if (p==1)
    {
        insertBeg();
    }
    else
    {
        printf("Enter the data of node : ");
        scanf("%d",&a);
        // to reach at one pre box of which to be inserted XD
        ptr = head;
        for(int i=1;i<p-1;i++)
        {
            ptr = ptr->next;
        }
        ptr2 = (box*)malloc(sizeof(box));
        ptr2->data = a;
        ptr2->next = ptr->next;
        ptr->next = ptr2;
    }
    //return *head;
}

struct node deleteBeg() //4 done
{   
    ptr = head; //old head is stored in ptr2
    while(ptr->next!=head){
        ptr=ptr->next; 
    } // now ptr will have location of last node
    ptr2 = head;
    head = ptr2->next; // now ptr2->next is new head
    ptr->next = head; // now last node will have add. of new head
    free(ptr2); //old head is now free 
}

struct node deleteEnd() //5 done
{
    ptr = head;
    while(ptr->next !=head)
    { 
      ptr2 = ptr; //since we need one previous node
      ptr = ptr->next; 
    } // now ptr will have location of last node
    ptr2->next = head; //2nd last node is connected to head
    free(ptr); // last node is free now
}

struct node deleteMid() //6 already done
{
    int p;
    printf("Enter the position of node to delete : ");
    scanf("%d",&p);
    if (p==1)
    {
        deleteBeg();
    }
    else
    {
        // to reach at one pre box of which to be deleted XD
        ptr = head;
        for(int i=1;i<p-1;i++)
        {
            ptr = ptr->next;
        }
        ptr2 = ptr->next;
        ptr->next = ptr2->next; //next k next wala kaam nahi kiya 
        free(ptr2);
    }
}


