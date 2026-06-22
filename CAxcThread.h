#ifndef _C_AXC_THREAD_H_
#define _C_AXC_THREAD_H_

#include <stdint.h>

typedef uint32_t(*axc_thread_proc)(void* pContext);

class CAxcThread
{
public:
	CAxcThread();
	virtual ~CAxcThread();

	bool IsValid() { return (m_hThread != NULL); }

	void* SetContext(void* pContext) { void* pResult = m_pContext; m_pContext = pContext; return pResult; }
	void* GetContext() { return m_pContext; }

	bool Create(
		axc_thread_proc pfnThreadProc,
		void* pContext,
		uint32_t dwStackSize = 0,
		int iPriority = 0);

	bool Destroy(
		uint32_t dwWaitMilliSeconds,
		bool bTerminateIfTimeout = true);

	bool CheckAlive(); 

	inline uint64_t GetThreadId() { return m_ddwThreadId; }

public:
	void OnGlobalCallback();

protected:
	void* m_hThread;
	uint64_t m_ddwThreadId;
	axc_thread_proc m_pfnThreadProc;
	void* m_pContext;
};


#endif // _C_AXC_THREAD_H_
