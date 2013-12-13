#ifndef _KTRACE_
#define	_KTRACE_

#include <stdlib.h>
#include <stdio.h>
#include <io.h>
#include <fcntl.h>
#include <process.h>
#include <math.h>
#include <string.h>
#include <stdarg.h>

namespace KTRACE
{
	enum PIPES { READ, WRITE }; /* Constants 0 and 1 for READ and WRITE */
	class KTrace
	{
	private:
		int fdpipe[2];
		int bufsize;
		int pid;
		char hstr[20];
		char sstr[20];
		char displayTool[1024];
		char *buf;
		bool success;
		int termstat;
		bool bad_read_client;
		bool running;

	public:
		KTrace(const char *ktrace_exe_path = "KTrace.exe",const int _bufsize = 10240)
		{
			bad_read_client = false;
			running = false;

			strcpy(displayTool, ktrace_exe_path);
			
			if((_access(displayTool, 0 )) == -1 ) {
				char *pPath = getenv("PATH");
				if(!pPath) return;

				int pathlen = strlen(pPath);
				
				char *pathbuf = new char[pathlen + 1];
				char *newPath = new char[2048];
				
				strcpy(pathbuf, pPath);
				char *dir = strtok(pathbuf, ";");
				bool findExeInPath = false;
				while(dir) {
					strcpy(newPath, dir);
					int len = strlen(newPath);
					if(len > 0 && !(newPath[len - 1] == '\\' || newPath[len - 1] == '/')) {
						strcat(newPath, "\\");
					}
					strcat(newPath, displayTool);
					if((_access(newPath, 0 )) != -1) {
						findExeInPath = true;
						strcpy(displayTool, newPath);
						break;
					}

					dir = strtok(NULL, ";");
				}

				delete []newPath;
				delete []pathbuf;

				if(!findExeInPath) return;
			}

			bufsize = _bufsize;
			success = (_pipe(fdpipe,bufsize,O_BINARY) != -1);
			if(success)
			{
				itoa(fdpipe[READ],hstr,10);
				itoa(bufsize,sstr,10);
				buf = (char *)malloc(sizeof(char) * bufsize);
			}
			pid = 0;
		}
		
		void Run()
		{
			if(success) {
				pid = _spawnl(P_NOWAIT,displayTool,displayTool,hstr,sstr,NULL);          
				if(pid != -1) {
					running = true;
				}
			}
		}

		void Trace(const char *fmt, ...)
		{
			if(running && !bad_read_client)
			{
				va_list ap;
				va_start(ap,fmt);
				vsprintf(buf,fmt,ap);
				if(_write(fdpipe[WRITE], buf, bufsize) == -1) {
					bad_read_client = true;
				}
				va_end(ap);
			}
		}

		~KTrace()
		{
			if(success)
			{
				Trace("!!EXIT!!");
				_cwait( &termstat, pid, WAIT_CHILD );
				_close( fdpipe[READ] );
				_close( fdpipe[WRITE] );
				free(buf);
			}
		}
	};
}


#endif