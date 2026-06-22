#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <memory.h>
#include <errno.h>
#include <stdint.h>
#include <pthread.h>
#include <errno.h>
#include <signal.h>
#include <fcntl.h>
#include <unistd.h>
#include <termios.h>
#include <netdb.h>
#include <dirent.h>
#include <fnmatch.h>
#include <stdarg.h>
#include <sys/time.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <net/if.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <ifaddrs.h>
#include <sys/types.h>
#include <sys/stat.h>

#include "CAxcThread.h"

typedef struct _AXC_T_THREAD_HANDLE
{
	pthread_t pt;
} AXC_T_THREAD_HANDLE;

static void* GlobalThreadProc(void* lpContext)
{
	CAxcThread* pThread = (CAxcThread*)lpContext;
	pThread->OnGlobalCallback();
	return NULL;
}

CAxcThread::CAxcThread()
{
	m_hThread = NULL;
	m_ddwThreadId = 0;
	m_pfnThreadProc = NULL;
	m_pContext = NULL;
}

CAxcThread::~CAxcThread()
{
	if (IsValid())
	{
		Destroy(0, false);
	}
}

bool CAxcThread::Create(axc_thread_proc pfnThreadProc, void* pContext, uint32_t dwStackSize, int iPriority)
{
	if (IsValid())
	{
		return false;
	}

	if (NULL == pfnThreadProc)
	{
		return false;
	}

	m_pfnThreadProc = pfnThreadProc;
	m_pContext = pContext;

	m_hThread = (void*)malloc(sizeof(AXC_T_THREAD_HANDLE));
	if (NULL == m_hThread)
	{
		m_pfnThreadProc = NULL;
		m_pContext = NULL;
		return false;
	}
	memset(m_hThread, 0, sizeof(AXC_T_THREAD_HANDLE));
	AXC_T_THREAD_HANDLE* pThreadHandle = (AXC_T_THREAD_HANDLE*)m_hThread;
	pThreadHandle->pt = 0;

	pthread_attr_t* pattr = NULL;
	pthread_attr_t attr;
	pthread_attr_init(&attr);

	if (0 != dwStackSize)
	{
		if (0 == pthread_attr_setstacksize(&attr, dwStackSize))
		{
			pattr = &attr;
		}
	}

	if (iPriority != 0 && iPriority >= -15 && iPriority <= 15)
	{
		if (0 == pthread_attr_setschedpolicy(&attr, SCHED_RR))
		{
			const int iMin = sched_get_priority_min(SCHED_RR);
			const int iMax = sched_get_priority_max(SCHED_RR);
			if (iMin < iMax)
			{
				int iNew = iMin + (iMax - iMin) * (iPriority + 15) / 30;
				if (iNew < iMin)
				{
					iNew = iMin;
				}
				else if (iNew > iMax)
				{
					iNew = iMax;
				}
				struct sched_param param;
				memset(&param, 0, sizeof(param));
				param.sched_priority = iNew;
				if (0 == pthread_attr_setschedparam(&attr, &param))
				{
					pattr = &attr;
				}
			}
		}
	}

	const int iResult = pthread_create(&pThreadHandle->pt, pattr, GlobalThreadProc, this);

	pthread_attr_destroy(&attr);

	if (0 != iResult)
	{
		m_pfnThreadProc = NULL;
		m_pContext = NULL;
		free(m_hThread);
		m_hThread = NULL;
		m_ddwThreadId = 0;
		return false;
	}

	m_ddwThreadId = pThreadHandle->pt;
	return true;
}

bool CAxcThread::Destroy(uint32_t dwWaitMilliSeconds, bool bTerminateIfTimeout)
{
	if (NULL == m_hThread)
	{
		return false;
	}

	if (CheckAlive())
	{
		AXC_T_THREAD_HANDLE* pThreadHandle = (AXC_T_THREAD_HANDLE*)m_hThread;
		if (0 != pThreadHandle->pt)
		{
			void* pt_rtn = NULL; 
			pthread_join(pThreadHandle->pt, &pt_rtn); 
			pthread_detach(pThreadHandle->pt);
			pThreadHandle->pt = 0;
		}
	}

	free(m_hThread);
	m_hThread = NULL;
	m_ddwThreadId = 0;

	return true;
}

bool CAxcThread::CheckAlive()
{
	if (NULL != m_hThread)
	{
		AXC_T_THREAD_HANDLE* pThreadHandle = (AXC_T_THREAD_HANDLE*)m_hThread;
		return (0 != pThreadHandle->pt);
	}
	return false;
}

void CAxcThread::OnGlobalCallback()
{
	if (m_pfnThreadProc)
	{
		m_pfnThreadProc(m_pContext);
	}
}
