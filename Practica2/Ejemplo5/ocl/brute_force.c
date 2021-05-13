#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include "md5.h"
#include "my_ocl.h"

#define MAX_DIGIT_LEN 10

int hex2int(char ch)
{
    if (ch >= '0' && ch <= '9')
        return ch - '0';
    if (ch >= 'A' && ch <= 'F')
        return ch - 'A' + 10;
    if (ch >= 'a' && ch <= 'f')
        return ch - 'a' + 10;
    return -1;
}

int main(int argc, char **argv)
{
	if (argc!=2){
		printf("./exec md5_hash[32chars]\n");
		exit(-1);
	}
	char *md5_hash = argv[1];
	size_t len = strlen(md5_hash);
	if (len!=32){
		printf("./exec md5_hash[32chars]\n");
		exit(-1);
	}

	char tmp[8];
	uint32_t h, h0_md5, h1_md5, h2_md5, h3_md5;
	for(int i=0; i<32; i+=8){
		h=0;
		for (int j=0; j<8; j++){ 
			char byte = md5_hash[i+j];
			char hex = hex2int(byte);
			h+=hex<<((7-j)*4);
		}
		if (i==0) h0_md5=h;
		if (i==8) h1_md5=h;
		if (i==16) h2_md5=h;
		if (i==24) h3_md5=h;
	}

	char cset[]="abcdefghijklmnopqrstuvwxyz 0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ\0";
	char msg_cracked[MAX_DIGIT_LEN];

	force_bruteOCL(msg_cracked, 5, cset, h0_md5, h1_md5, h2_md5, h3_md5);
	
		printf("md5=%s corresponds to msg=%s\n", md5_hash, msg_cracked);
}
