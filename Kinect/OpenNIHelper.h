#pragma once

#include <XnCppWrapper.h>
#include <iostream>
#include <string>

class COpenNIHelper {
public:
	xn::Context mContext;
	xn::DepthGenerator mDepthGenerator;
	xn::ImageGenerator mImageGenerator;
	xn::UserGenerator mUserGenerator;

	xn::Player mPlayer;
	
private:
	XnStatus eResult;
	bool inited;
	

#define CheckOpenNIError(e)     \
		if(e != XN_STATUS_OK) { \
			std::cerr << __LINE__ << " Error: " << xnGetStatusString( e ) << std::endl; \
			return false; \
		}
public:
	COpenNIHelper() {
		eResult = XN_STATUS_OK;
		inited = false;
	}

	~COpenNIHelper() {
		mContext.StopGeneratingAll();
		mContext.Shutdown();
	}

	bool InitOpenNI() {
		eResult = mContext.Init();
		CheckOpenNIError(eResult);
		
		mContext.SetGlobalMirror(false);

		// set map mode
		XnMapOutputMode mapMode;
		mapMode.nXRes = 640;
		mapMode.nYRes = 480;
		mapMode.nFPS = 30;

		//create image and depth generator
		CheckOpenNIError( mDepthGenerator.Create(mContext) );
		CheckOpenNIError( mDepthGenerator.SetMapOutputMode(mapMode) );
		CheckOpenNIError( mImageGenerator.Create(mContext) );
		CheckOpenNIError( mImageGenerator.SetMapOutputMode(mapMode) );
		CheckOpenNIError( mUserGenerator.Create(mContext) );
		//correct view port
		mDepthGenerator.GetAlternativeViewPointCap().SetViewPoint( mImageGenerator );
		
		// 2. User generator
		xn::SkeletonCapability skel = mUserGenerator.GetSkeletonCap();

		// 3. PoseDetection & Skeleton
		float m_SmoothingFactor = 0.1;
		CheckOpenNIError( skel.SetSmoothing(m_SmoothingFactor) );
		CheckOpenNIError( skel.SetSkeletonProfile(XN_SKEL_PROFILE_ALL) );
		
		//start generate data
		CheckOpenNIError( mContext.StartGeneratingAll() );
		inited = true;
		return true;
	}

	bool InitOpenNiwithONI(const char *record_path) {
		CheckOpenNIError( mContext.Init() );
		CheckOpenNIError( mContext.OpenFileRecording(record_path, mPlayer) );
		CheckOpenNIError( mContext.FindExistingNode(XN_NODE_TYPE_DEPTH, mDepthGenerator) );
		CheckOpenNIError( mContext.FindExistingNode(XN_NODE_TYPE_IMAGE, mImageGenerator) );
		mDepthGenerator.GetAlternativeViewPointCap().SetViewPoint( mImageGenerator );
		CheckOpenNIError( mUserGenerator.Create(mContext) );
		inited = true;

		// 2. User generator
		xn::SkeletonCapability skel = mUserGenerator.GetSkeletonCap();

		// 3. PoseDetection & Skeleton
		float m_SmoothingFactor = 0.1;
		CheckOpenNIError( skel.SetSmoothing(m_SmoothingFactor) );
		CheckOpenNIError( skel.SetSkeletonProfile(XN_SKEL_PROFILE_ALL) );
		
		CheckOpenNIError( mContext.StartGeneratingAll() );

		return true;
	}

	bool CleanKinect() {
		if(!inited) return true;

		eResult = mContext.StopGeneratingAll();
		CheckOpenNIError(eResult);

		mContext.Shutdown();
		inited = false;
		return true;
	}
};