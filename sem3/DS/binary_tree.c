#include<stdio.h>
#include<time.h>
#include<stdlib.h>

int myrandom(int lower=0,int upper=100)
{  
    int num = (rand() % (upper-lower +1)) + lower;
    // srand(time(0));
    return num;
}
typedef struct node
{
    int data;
    struct node *left;
    struct node *right;
}node;
node *root,*left_child,*right_child,*baap,*temp1,*temp2;

struct node insert(node *temp);

/*-----------------------------------*/

int main(void)
{   
    int n=5; //
    
    srand(time(0));
    root = (node*)malloc(sizeof(node));
    root->data = myrandom();
    root->left = NULL;
    root->right = NULL;
    temp1 = root;
    temp2 = root;
    for(int i=0;i<n;i++)
    {
        insert(temp1);
    }
    return 0;
}
struct node insert(node *temp)
{
    left_child = (node*)malloc(sizeof(node));
    left_child->data = myrandom();
    left_child->left = NULL;
    left_child->right = NULL;
    temp1->left = left_child; // roo k left me
    temp1 = left_child;
    //insert(temp1); 

    right_child = (node*)malloc(sizeof(node));
    right_child->data = myrandom();
    right_child->left = NULL;
    right_child->right = NULL;
    temp2->right = right_child; // root k right me 
    temp2 = right_child;
    //insert(temp2);
}