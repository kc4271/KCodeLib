#include <stdlib.h>
#include <stdio.h>
#include <io.h>
#include <fcntl.h>
#include <process.h>
#include <math.h>
#include <string.h>

enum PIPES { READ, WRITE }; /* Constants 0 and 1 for READ and WRITE */
int hpipe[2];
int bufferSize;

int main(int argc, char **argv) {
	if(argc < 3) {
		printf("This program need [pipe read id] and [buffer size]\n");
		return 0;
	}

	hpipe[READ] = atoi(argv[1]);
	bufferSize = atoi(argv[2]);

	const char *terminal = "!!EXIT!!";
	int len = strlen(terminal);

	char *buf = new char [bufferSize];

	for(;;)
	{
		_read( hpipe[READ], buf, bufferSize);
		
		if(!strncmp(buf, terminal, len)) {
			break;
		}
		printf("%s", buf);
	}

	delete []buf;

	return 0;
}