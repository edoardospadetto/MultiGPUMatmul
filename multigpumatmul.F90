
!****************************************
#define GPUZMPLX
!***************************************
!Include Modules

#include "modules/debugqic.F90"
#include "modules/matrixqic.F90"

!******** MACROS IMPLEMENTATIONS ****************
!Macros are valid only inside Parallel Module, the main program must be type specific 
#ifdef GPUREAL
#define CTYPE	REAL
#define CUBLAS_TEMPLATE_GEMM  CUBLAS_SGEMM
#define CTYPE1 1.0
#define CTYPE0 0.0
#endif

#ifdef GPUCMPLX
#define CTYPE complex
#define CUBLAS_TEMPLATE_GEMM  CUBLAS_CGEMM
#define CTYPE1 cmplx(1.0,0.0)
#define CTYPE0 cmplx(0.0,0.0)
#endif

#ifdef GPUZMPLX
#define CTYPE double complex
#define CUBLAS_TEMPLATE_GEMM  CUBLAS_ZGEMM
#define CTYPE1 dcmplx(1.0,0.0)
#define CTYPE0 dcmplx(0.0,0.0)
#endif

!***********************************************


module parallel
	USE OMP_LIB 
	
	implicit none
	contains 
	!*Debug subroutine 4 the most used cublas routines
	subroutine check(val, quit)
		integer:: val
		logical:: quit
		if (val .ne. 0) then
			quit = .true. 
			write(*,*) "Cublas Memory Error"
		end if
	
	end subroutine
	
	!########COMPUTES SIZES OF THE chunkS TO SUBMIT TO THE GPU###########
	subroutine select_chunks_size(sizes,free_mem,bytes, fractions)
		real , intent(IN) :: free_mem
		integer*8,intent(IN) :: bytes
		integer,dimension(3), intent(INOUT):: sizes 
		integer, dimension(3), intent(INOUT) ::fractions 
		integer, dimension(3) :: sizes_tmp
		integer :: gpuload
		integer*2 :: ii=1 , jj
		fractions = (/1,1,1/)
		sizes_tmp = sizes
		gpuload = bytes*(sizes_tmp(1)*sizes_tmp(2)+sizes_tmp(2)*sizes_tmp(3)+sizes_tmp(1)*sizes_tmp(3))
		
		
		!Dummy algorithm can be improved
		
		 do
			 if (gpuload .lt. free_mem) then
			 	EXIT
			 end if
		 
			fractions(ii) = fractions(ii)+1 
			do jj = 1, 3
				sizes_tmp(jj) = int ( ceiling ( real( sizes(jj) ) /  real(fractions(jj)) ) )
			end do 
			
			gpuload = bytes*(sizes_tmp(1)*sizes_tmp(2)+sizes_tmp(2)*sizes_tmp(3)+sizes_tmp(1)*sizes_tmp(3))
			!mems = bytes of a number * (colschunkA*rowschunkA + colschunkB*rowschunkB +colschunkC*rowschunkC)
			
			ii = ii+1
			if (ii .eq. 4) then
				ii = 1
			end if
		end do 
		
		sizes=sizes_tmp
		
		
	end subroutine
	!###################
	
	!#################################################	
	!MATMUL
	! A,B are input matrices, C is the result. ngpu
	!is the number of gpu to use
	!gpugrid_in is a 2d array that choose how to 
	!shape the blocks of the C matrix
	subroutine pmatmul(A, B, C,ngpu, gpugrid_in)
		implicit none
		
		!CUBLAS HELPER FUNCTIONS
		!Respect c++ return types
		integer cublas_init 
		integer cuda_set_device
		integer cuda_get_device
		integer cuda_get_device_count
		integer cuda_mem_get_info
		integer CUBLAS_TEMPLATE_GEMM
		integer cublas_set_matrix
		integer cublas_get_matrix
		integer cublas_alloc
		integer cublas_free
		
		
		!Input Output Instances
		integer, intent(IN) :: ngpu
		integer, dimension(2), intent(INOUT), optional :: gpugrid_in
		CTYPE, dimension(:,:), intent(IN) :: A, B 
		CTYPE, dimension(:,:), intent(INOUT) :: C
		
		!Thread side parameters
		integer, dimension(2) :: gpugrid
		integer, dimension(2) :: blocksdim  
		integer, dimension(2) :: blockscoord
		integer, dimension(2) :: blocksopcols(2), blocksoprows(2)
		
		!GPU side parameters
		!cuda
		integer :: detected_gpus
		real :: free_mem, total_mem
		!cublas
		logical :: quitcublas =.false.
		integer :: cublas_stat
		integer(8) :: devPtrA, devPtrB, devPtrC
		!algorithm
		integer:: times(3), chunk_sizes(3) ! dimensions which specifies how many bytes to load in the gpu
		integer*2 :: ii,jj,kk
		integer :: chunk_lims(6), deltas(3)
		CTYPE , dimension(:,:), allocatable :: tmp_result
		
		
		
		
		
		!#### CHECKS ####
		C=0
		!Divide result Matrix in N blocks as the number of GPU.
		gpugrid = (/1,ngpu/)
		if( present(gpugrid_in) ) then 	
			if (gpugrid_in(1)*gpugrid_in(2) .eq. ngpu ) then
				gpugrid = gpugrid_in
			else 
				write(*,*) "Bad Process Grid, Using default Grid"
			end if
		else 
			
			write(*,*) "Missing Process Grid, USing Default Grid"

		end if 
		 		 
		!Check valid C array 
		if ( (size(A,2) .ne. size(B,1)) .or. (size(A,1) .ne. size(C,1))  .or. (size(B,2) .ne. size(C,2)) ) then 
			write(*,*) "ERROR : Not valid dimensions of matrices, killing program"
			STOP
		end if 
		
		!Check number of GPU.
		if(cuda_get_device_count(detected_gpus) .ne. 0) then 
			write (*,*) "Cuda Error when counting devices"
			STOP
		end if
		
		if(detected_gpus .lt. ngpu) then 
			write (*,*) "Error, requested invalid number of gpus"
			!STOP
		end if 
		
		
		!#### PREPROCESSING ##### Thread side
		!Divide result matrix in blocks and let each thread compute one.
		
		!Compute blocks dimension.
		blocksdim(1)= int ( ceiling ( real( size(C,1) ) /  real( gpugrid(1) ) ) ) 
		blocksdim(2)= int ( ceiling ( real( size(C,2) ) /  real( gpugrid(2) ) ) ) 
		
		call omp_set_num_threads(ngpu)
						
		!$OMP PARALLEL DEFAULT(PRIVATE) SHARED(A,B,C,gpugrid,ngpu,blocksdim) 
		
		
		!Coordinates of the thread in the process grid
		
		blockscoord(1)= (OMP_GET_THREAD_NUM()) / gpugrid(2)  +1
		blockscoord(2)=  mod(OMP_GET_THREAD_NUM(), gpugrid(2)) +1
		
		!Associate portion of the result matrix to a thread.
		
		blocksoprows(1)=(blockscoord(1)-1)*blocksdim(1) +1  
		blocksoprows(2)=blockscoord(1)*blocksdim(1) 
		blocksopcols(1)=(blockscoord(2)-1)*blocksdim(2) +1 
		blocksopcols(2)=blockscoord(2)*blocksdim(2) 
		
		
		!Check threads at the bounds.
		if(blockscoord(1) .eq. gpugrid(1)) then 
			blocksoprows(2)=  size(C,1)
		end if
		if (blockscoord(2) .eq. gpugrid(2)) then 
			blocksopcols(2)=  size(C,2)
		end if		
		
		!########### PREPROCESSING ########## GPU SIDE
		if (cublas_init() .ne. 0) then 
			write (*,*) "Cublas Init Failed" 
			STOP
		end if 
		
		!set the right devices 
		 if (cuda_set_device(OMP_GET_THREAD_NUM()) .ne. 0 ) then 
		 	write(*,*) "Error setting device" 
		 	call cublas_shutdown()	
		 	STOP
		end if	
		
		!Check GPU MEMORY 
		if(CUDA_MEM_GET_INFO(free_mem ,total_mem) .ne. 0 ) then 
			write (*,*) "Cuda Error when checking memory"
			call cublas_shutdown()	
			STOP
		end if  

		
		!Compute chunks of matrices
		
		chunk_sizes(1) = blocksoprows(2)-blocksoprows(1)+1 !rows of A / rows of C ---- +1 due to fortran indexing
		chunk_sizes(2) = blocksopcols(2)-blocksopcols(1)+1 !cols of B  /cols of C
		chunk_sizes(3) = size(A,2)			   !cols of A / rows of B
		
		call select_chunks_size(chunk_sizes,free_mem,sizeof(A(1,1)), times) ! return the sizes of the chunks to submit to the gpus
		
		write(*,*) times
		
		!########## MATRIX MULTIPLICATION ##################
		
		!write(*,*) OMP_GET_THREAD_NUM() , times
		do ii = 1, times(1)  
			do jj = 1, times(2)
				do kk = 1, times(3)
					!Chunk Lims represent the bounds of the chunks of A,B,C submitted to the gpu
					!deltas are the dimensions of the chucnks
					!A rows, C rows
						chunk_lims(1) =  blocksoprows(1) + chunk_sizes(1)*(ii-1) !+1 induced by fortran indexing is 
						chunk_lims(2) =  blocksoprows(1) + chunk_sizes(1)*ii -1  !inside blocksoprow
					!B cols,  C cols
						chunk_lims(3) =  blocksopcols(1)+chunk_sizes(2)*(jj-1)  
						chunk_lims(4) =  blocksopcols(1)+chunk_sizes(2)*jj -1
					!A cols, B rows 
						chunk_lims(5) =  chunk_sizes(3)*(kk-1)+1 
						chunk_lims(6) =  chunk_sizes(3)*kk
						
					!If statement to do not trepass the block bounds (caused by ceiling)
					if(ii .eq. times(1)) then
						chunk_lims(2) =  blocksoprows(2)
					end if
					
					if(jj .eq. times(2)) then
						chunk_lims(4) =  blocksopcols(2)
					end if
					
					if(kk .eq. times(3)) then 
						chunk_lims(6) = size(A,2)
					end if
					!write(*,*) ii,jj,kk
					
					deltas(1)= chunk_lims(2)-chunk_lims(1)+1
					deltas(2)= chunk_lims(4)-chunk_lims(3)+1
					deltas(3)= chunk_lims(6)-chunk_lims(5)+1
					
					allocate(tmp_result(deltas(1), deltas(2)))
					
					!**********
					!write(*,*) "-----"
					!write(*,*) OMP_GET_THREAD_NUM(), "|", chunk_lims,"|", blocksoprows, "|", blocksopcols
					!write(*,*) "-----"
					!**********
					
					quitcublas =.false.
					write (*,*) sizeof(A(1,1)), deltas
					call check( cublas_alloc(deltas(1)*deltas(3), sizeof(A(1,1)), devPtrA), quitcublas)
					call check( cublas_alloc(deltas(3)*deltas(2), sizeof(A(1,1)), devPtrB), quitcublas)
					call check( cublas_alloc(deltas(1)*deltas(2), sizeof(A(1,1)), devPtrC), quitcublas)
					
					 if (quitcublas) then 
					 	write(*,*) "Cublas Alloc Error" 
					 	call cublas_shutdown()	
					 	STOP
					 end if	
					
					!copy data to GPU
					call check(cublas_set_matrix(deltas(1),deltas(3),sizeof(A(1,1)),&
						A(chunk_lims(1):chunk_lims(2), chunk_lims(5):chunk_lims(6)),deltas(1),devPtrA,deltas(1)), quitcublas)  
					call check(cublas_set_matrix(deltas(3),deltas(2),sizeof(A(1,1)),&
						B(chunk_lims(5):chunk_lims(6), chunk_lims(3):chunk_lims(4)),deltas(3),devPtrB,deltas(3)), quitcublas)  
					call check(cublas_set_matrix(deltas(1),deltas(2),sizeof(A(1,1)),tmp_result,deltas(1),devPtrC,deltas(1)), quitcublas) 
					
					 if (quitcublas) then 
					 	write(*,*) "Cublas Set Matrix Error" 
					 	call cublas_shutdown()	
					 	STOP
					 end if	
					 
			
					
					cublas_stat = CUBLAS_TEMPLATE_GEMM ('N', 'N', &
					   deltas(1), &  !rows chunk A chunk C
					   deltas(2), &  !cols chunk B chunk C
					   deltas(3), &  !rows chunk B  cols chunk A
					   CTYPE1, & !alpha parameter
					   devPtrA, & 
					   deltas(1), &
					   devPtrB,  & 
					   deltas(3), &
					   CTYPE0, & !beta parameter
					   devPtrC, &
					   deltas(1)) 
					  
					  call check(cublas_get_matrix(deltas(1),deltas(2),sizeof(A(1,1)),&
					  	devPtrC,deltas(1),tmp_result,deltas(1)),quitcublas)
					
					
					 C(chunk_lims(1):chunk_lims(2), chunk_lims(3):chunk_lims(4)) = & 
					 	 C(chunk_lims(1):chunk_lims(2), chunk_lims(3):chunk_lims(4)) + tmp_result
					 
					 deallocate(tmp_result)
					     !Free GPU memory
					
					   
					 if (cublas_stat .ne. 0 ) then 
					 	write(*,*) "Matmul Error" 
					 	call cublas_shutdown()	
					 	STOP
					 end if		
					 
					call check(cublas_free(devPtrA),quitcublas)
					call check(cublas_free(devPtrB),quitcublas)
					call check(cublas_free(devPtrC),quitcublas)
					
					 if (quitcublas) then 
					 	write(*,*) "Cublas Free Error" 
					 	call cublas_shutdown()	
					 	STOP
					 end if	
		
				end do 
			end do 
		end do 
		
		 
		
		
		
		
		call cublas_shutdown()	
		!##############################################
		!$OMP END PARALLEL
		
		
		return	
		
	
	end subroutine
	
	
	


end module 

program mutml
	USE OMP_LIB 
	use parallel
	use matrixqic
	use debugqic
	CTYPE :: A(30,40), B(40,30), C(30,30)
	integer, dimension(2):: grid = (/2,2/) 
	write(*,*)"-----"
	A=rgzm(30,40)*675
	B=rgzm(40,30)*132
	write(*,*)"-----"
	!print input matrix A
	!call pcm(A)
	write(*,*)"-----"
	!call pcm(b)
	
	call pmatmul(A, B, C,1 )
	write(*,*)"----- Fortran Matmul ---"
	!call pcm(matmul(A,B))
	write(*,*)"----- My result ------"
	!call pcm(C)
	write(*,*)"----- CHECK -----"
	write(*,*)"sum(Matmul(A,B), custommatmul(A,B,gpu,etc..)) =" ,sum(matmul(A,B)-C)
	
	
 	
end program 
