/*
 * File description: manual_reset_event.h
 * Author information: Mike Ranzinger mranzinger@alchemyapi.com
 * Copyright information: Copyright Orchestr8 LLC
 */

#ifndef MANUAL_RESET_EVENT_H_
#define MANUAL_RESET_EVENT_H_

#include <mutex>
#include <condition_variable>

class event
{
private:
	std::mutex _lock;
	std::condition_variable _cond;
	bool _signalled;
	bool _manual;

public:
	event(bool manual = true) : _manual(manual), _signalled(false) { }

	void set()
	{
		{
			std::lock_guard<std::mutex> lock(_lock);
			_signalled = true;
		}

		_cond.notify_all();
	}

	void reset()
	{
		std::lock_guard<std::mutex> lock(_lock);
		_signalled = false;
	}

	void wait()
	{
		std::unique_lock<std::mutex> lock(_lock);
		while (!_signalled)
		{
			_cond.wait(lock);
		}
		// Reset the signal now that it has been acquired.
		// We also own the lock right now, so everything is safe
		_signalled = false;
	}
};



#endif /* MANUAL_RESET_EVENT_H_ */
