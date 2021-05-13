__kernel
void calc_Pi_kernel (int n, __global double * acc)
{
	int a = get_global_id (0);
	if (a < n)
	{
		double x;
		x = ((a + 1) + 0.5) / n;
		acc[a] = 4.0 / (1.0 + x*x);
	}
}
