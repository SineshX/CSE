#include<stdio.h>
#include<stdlib.h>

typedef struct node  
{
  int data;
  struct node *next;
}node;

int main()
{
  int n,i=1,a;
  node *head1,*head,*ptr,*ptr2,*ptr3,*ptrz;
  printf("Enter the no. of nodes ");
  scanf("%d",&n);

  printf("Enter the %d node ",i);
  scanf("%d",&a);
  ptr = (node*)malloc(sizeof(node));
  
  ptr->data = a;
  ptr->next = NULL;
  head = ptr;
  head1=ptr;

  for(i=2;i<=n;i++)
    {
      printf("Enter the %d node ",i);
      scanf("%d",&a);
      ptr2 = (node*)malloc(sizeof(node));
      ptr2->data = a;
      ptr2->next = NULL;
      ptr->next = ptr2;
      ptr = ptr->next;
    }

  ptr = head;
  while(ptr != NULL)
  {
    printf("  %d\t",ptr->data);//printing
    ptr = ptr->next;
  }

    ptr = head;
    while(ptr->data!=20)
    {
        ptr = ptr->next;
    }

    ptr3 = (node*)malloc(sizeof(node));
    ptr3->data = 30;
    ptr3->next = ptr->next;
    ptr->next = ptr3;


    ptr = head;
    while(head != NULL)
  {
    printf("  %d\t",ptr->data);
    head = head->next;
  }


}