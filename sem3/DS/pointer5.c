/* segvhunt.c
 Find and eliminate all code that generates Segmentation Fault
*/
#include <stdio.h>
int main() {
 char **s;
 char foo[] = "Hello World";
   //printf("%s\n%d",foo,sizeof(foo));
 *s = foo;
 printf("s is %s\n",s);
 s[0] = foo;
 printf("s[0] is %s\n",s[0]);
 return(0);
}