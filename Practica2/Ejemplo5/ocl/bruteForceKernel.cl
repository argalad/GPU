#define MAX_DIGIT_LEN 10

__kernel
void bruteForce (__global char msg_cracked[], int nchars, __global char cset[],
				 int cset_len, uint h0_md5, uint h1_md5, uint h2_md5, uint h3_md5)
{
	int idx = get_global_id(0);
	int idy = get_global_id(1);

	char msg[MAX_DIGIT_LEN];
	int basis[MAX_DIGIT_LEN];

	int acc = 1;
	for(int i = 0; i < MAX_DIGIT_LEN; i++)
    {
		acc *= cset_len;
		basis[i] = acc;
	}

	int nostop = 1;
	uint h0, h1, h2, h3;

	printf ("Trying m5d with %d characters!!\n", nchars+1);

	if (idx < nchars && idy < basis[idx])
	{
		//gen_msg (msg, nchar, cset, basis, ichar);
		msg[idx+1] = '\0';
		for (int j = 0; j <= idx; j++)
		{
			if (j > 0)
				msg[idx-j] = cset[(idy % basis[j]) / basis[j-1]];
			else
				msg[idx-j] = cset[idy % basis[j]];
		}

		md5 (&h0, &h1, &h2, &h3, msg, strlen (msg));
		//printf("value = %s(%d) %x %x %x %x\n", msg, strlen(msg), h0, h1, h2, h3);

        nostop = (h0 == h0_md5)&(h1 == h1_md5)&(h2 == h2_md5)&(h3 == h3_md5); //h0, h1, h2, h3 comparison
		if (nostop)
		{
			strcpy (msg_cracked, msg);
			//return (1);
		}
	}

	//return(0);
}