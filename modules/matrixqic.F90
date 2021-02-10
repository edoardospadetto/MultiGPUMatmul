

module matrixqic
    use debugqic
    !Some useful functions for handling matrices

    contains
    !%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    !generate random complex double matrix hermitian
    FUNCTION rghcm(tsize) RESULT(gen)
    !%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      IMPLICIT NONE
      INTEGER, INTENT(IN) :: tsize
      INTEGER ::  ii, jj
      DOUBLE precision :: a, b
      DOUBLE COMPLEX, DIMENSION(tsize,tsize) :: gen

      do ii= 1, tsize
          do jj = 1, ii
              call RANDOM_NUMBER(a)
              call RANDOM_NUMBER(b)
              gen(ii,jj) = cmplx(a,b,kind(0D0))
              if (ii .NE. jj) then
                  gen(jj,ii) = cmplx(a,-b,kind(0D0))
              end if
          end do
      end do
  END FUNCTION rghcm

  !%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  !generate random complex double matrix
  FUNCTION rgzm(sizea,sizeb) RESULT(gen)
  !%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    IMPLICIT NONE
    INTEGER, INTENT(IN) :: sizea,sizeb
    INTEGER ::  ii, jj
    DOUBLE precision :: a(sizea,sizeb), b(sizea,sizeb)
    DOUBLE COMPLEX, DIMENSION(sizea,sizeb) :: gen
    call RANDOM_NUMBER(a)
    call RANDOM_NUMBER(b)
    gen = dcmplx(a,b)
    !do ii= 1, sizea
    !    do jj = 1, sizeb
    !        call RANDOM_NUMBER(a)
    !        call RANDOM_NUMBER(b)
    !        gen(ii,jj) = cmplx(a,b,kind(0D0))
    !    end do
    !end do
END FUNCTION rgzm

  !%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  !generate random complex double matrix
  FUNCTION rgcm(sizea,sizeb) RESULT(gen)
  !%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    IMPLICIT NONE
    INTEGER, INTENT(IN) :: sizea,sizeb
    INTEGER ::  ii, jj
    real :: a(sizea,sizeb), b(sizea,sizeb)
    COMPLEX, DIMENSION(sizea,sizeb) :: gen
    call RANDOM_NUMBER(a)
    call RANDOM_NUMBER(b)
    gen = dcmplx(a,b)

    !do ii= 1, sizea
    !    do jj = 1, sizeb
    !        call RANDOM_NUMBER(a)
    !        call RANDOM_NUMBER(b)
    !        gen(ii,jj) = cmplx(a,b,kind(0D0))
    !    end do
    !end do
END FUNCTION rgcm
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!generate random double diagonal matrix
FUNCTION rgddm(sizem) RESULT(gen)
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  IMPLICIT NONE
  INTEGER, INTENT(IN) :: sizem
  INTEGER ::  ii, jj
  DOUBLE precision :: a
  DOUBLE precision, DIMENSION(sizem,sizem) :: gen

  do ii= 1, sizem
      do jj = 1, sizem
          gen(ii,jj) =0.0
      end do
      call RANDOM_NUMBER(a)
      gen(ii,ii) = a
  end do
END FUNCTION rgddm


    !%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    !generates random real matrix
    FUNCTION rgrm(sizerow,sizecol) RESULT(gen)
    !%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      IMPLICIT NONE
      INTEGER, INTENT(IN) :: sizerow,  sizecol
      REAL, DIMENSION(sizerow,sizecol) :: gen
      call RANDOM_NUMBER(gen)
  END FUNCTION rgrm

    !%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    !print a matrix on terminal
    SUBROUTINE prm(mat)
    !%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      IMPLICIT NONE
      !inout vars
      REAL, DIMENSION(:,:), INTENT(IN) :: mat
      !tmp vars
      INTEGER*2 :: ii,jj
      INTEGER, DIMENSION(SIZE(SHAPE(mat))) :: sizem

      sizem=SHAPE(mat)

      PRINT*, ""
      DO ii=1,sizem(1)
        DO jj=1,sizem(2)
          WRITE(*, fmt="(f0.2, tr2)", advance="no") mat(ii,jj)
        END DO
        PRINT*, ""
      END DO

  END SUBROUTINE prm

    !%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    !read dimension from file.
    FUNCTION read_dim(path) RESULT(dim)
    !%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    character(len = *) :: path
    integer :: dim
    logical :: file_exist = .FALSE.
    INQUIRE(FILE=path, EXIST=file_exist)

    call breakifn(path // "does not exists",file_exist, .TRUE.)

    open(unit = 3, file = path , status = "old" )

    read(3, fmt= "(i8)") dim

    close(3)

    END FUNCTION

!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!print a complexmatrix*8 on terminal
SUBROUTINE pzm(mat)
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   IMPLICIT NONE
   INTEGER :: ii,jj
   DOUBLE COMPLEX, DIMENSION(:,:), INTENT(IN) :: mat
   INTEGER, DIMENSION(SIZE(SHAPE(mat))) :: sizem
   sizem=SHAPE(mat)
   print*, "ok"
   PRINT*, ""
   DO ii=1,sizem(1)
        write(*, "(*(' 'sf6.2xspf6.2x'i ':x))")  mat(ii,:)
     PRINT*, ""
   END DO


END SUBROUTINE pzm

!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!print a complexmatrix on terminal
SUBROUTINE pcm(mat)
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   IMPLICIT NONE
   INTEGER :: ii,jj
   COMPLEX, DIMENSION(:,:), INTENT(IN) :: mat
   INTEGER, DIMENSION(SIZE(SHAPE(mat))) :: sizem
   sizem=SHAPE(mat)

   PRINT*, ""
   DO ii=1,sizem(1)
        write(*, "(*(' 'sf6.2xspf6.2x'i ':x))")  mat(ii,:)
     PRINT*, ""
   END DO


END SUBROUTINE pcm




    !%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    !compute eigenvalues and eigenvector with zheev,
    subroutine eigz(A,dimA,ev)
    !%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
           external zheev
           double complex, dimension(:,:), intent(INOUT) :: A
           double precision, dimension(:), intent(INOUT):: ev
           integer ::dimA,ii
           integer :: LWMAX, LWORK,INFO
           parameter(LWMAX = 1000000)
           complex*16 :: WORK(LWMAX)
           double precision :: RWORK(3*dimA-2)

           call breakifn("size eigenvalue not equal dimA  .. eigc:matrixqic ", size(ev, dim=1) .eq. dimA)
           call breakifn("size eigenvalue not equal dimA  .. eigc:matrixqic ", size(A, dim=1) .eq. dimA)
           call breakifn("size eigenvalue not equal dimA  .. eigc:matrixqic ", size(A, dim=2) .eq. dimA)

           LWORK = -1


            call zheev ( 'V','U',dimA,A,dimA,ev,WORK,LWORK,RWORK,INFO)

            !PRINT*, INT(WORK(1))
            LWORK = MIN(LWMAX, INT(WORK(1)))
            call zheev ( 'V','U',dimA,A,dimA,ev,WORK,LWORK,RWORK,INFO)
            call breakifn("bhoo", info .eq. 0, .TRUE.)



            !print*, "eig stuff computed"



    end subroutine eigz
    !##############################################
    !computes kroeneker product of double complex matrices
    function zkrcprod(A,B) result(C)
    !######################################
        integer :: sizea
        double complex, dimension(:,:) :: A
        double complex, dimension(:,:) :: B
        double complex, dimension(size(A, dim=1)*size(B, dim=1),size(A, dim=2)*size(B, dim=2)) :: C
        integer ::  rows(2), columns(2)
        do ii = 0, size(A, dim = 1)-1
            do jj = 0, size(A, dim = 2)-1

                rows(1) = size(B,dim=1)*ii+1
                rows(2) = size(B,dim=1)*ii+size(A, dim = 1)
                columns(1) = size(B,dim=2)*jj+1
                columns(2) = size(B,dim=2)*jj+size(A, dim = 2)
                !print*, rows(1), rows(2), columns(1), columns(2)
                C(rows(1):rows(2) , &
                  columns(1): columns(2)) = &
                     B * A(ii+1, jj+1)
                end do
            end do

    end function

end module
