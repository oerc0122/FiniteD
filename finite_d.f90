module finite_d
  implicit none

  integer, parameter :: dp = selected_real_kind(15,300)

contains

  ! subroutine finite_d_coeffs_points(points, derivOrder, direction)

  subroutine finite_d_coeffs_stencil( stencil, deriv, direction, points, coeffs, x0, store, storeN)
    integer,                                         intent( in    ) :: stencil
    integer,                                         intent( in    ) :: deriv
    integer,                                         intent( in    ) :: direction
    real(kind = dp), dimension(:), allocatable,      intent(   out ) :: points
    real(kind = dp), dimension(:), allocatable,      intent(   out ) :: coeffs
    real(kind = dp), dimension(:,:,:), allocatable :: coeffs_work

    real(kind = dp), intent( in    ), optional :: x0
    logical,         intent( in    ), optional :: store
    integer,         intent( in    ), optional :: storeN

    real(kind = dp) :: x0val
    logical :: storeval
    integer :: storenval

    integer :: stenMax

    real(kind = dp) :: c1, c2, c3
    integer :: i, n, v, m
    integer, dimension(3) :: ierr

    if (present(x0)) then
      x0val = x0
    else
      x0val = 0.0_dp
    end if

    if (deriv < 1) stop "Derivative order must be greater than or equal to one"

    if (2*stencil + 1 .lt. deriv .and. direction .eq. 0) then
      stop "Number of sample points must be larger than derivative"
    else if ( stencil + 1 .lt. deriv .and. direction .ne. 0) then
      stop "Number of sample points must be larger than derivative"
    end if

    select case (direction)
    case (:-1) ! Backward difference
      allocate(points(0:stencil), stat = ierr(1) )
      points = [ (real(i, dp), i = -stencil, 0) ]
      allocate(coeffs_work( 0:deriv, 0:stencil, 0:stencil ), stat = ierr(2) )
      allocate(coeffs( 0:stencil ), stat = ierr(3) )
    case (0)   ! Centred difference
      stenMax = 2*stencil
      allocate(points(0:stenMax), stat = ierr(1) )
      points = [ (real(i, dp), i = -stencil, stencil) ]
      allocate(coeffs_work( 0:deriv, 0:stenMax, 0:stenMax ), stat = ierr(2) )
      allocate(coeffs( 0:stenMax ), stat = ierr(3) )
    case (1:)  ! Forward difference
      allocate(points(0:stencil), stat = ierr(1) )
      points = [ (real(i, dp), i = 0, stencil) ]
      allocate(coeffs_work( 0:deriv, 0:stencil, 0:stencil ), stat = ierr(2) )
      allocate(coeffs( 0:stencil ), stat = ierr(3) )
    end select

    if (any(ierr .ne. 0)) &
      &      stop "Allocation error in finite_d_coeffs_stencil"

    coeffs_work = 0

    coeffs_work(0,0,0) = 1
    c1 = 1.0_dp

    nloop:do n = 1, size(points) - 1
      c2 = 1.0_dp

      vloop:do v = 0, n - 1
        c3 = points(n) - points(v)
        c2 = c2 * c3

        do m = 0, min(n, deriv)
          i = modulo(m-1, deriv)
          coeffs_work(m,n,v) = (( points(n) - x0val ) * coeffs_work(m,n-1,v) - m * coeffs_work(i,n-1,v)) / c3
        end do
      end do vloop

      do m = 0, min(n, deriv)
        i = modulo(m-1, deriv)
        coeffs_work(m,n,n) = (c1/c2) * (m * coeffs_work( i,n-1,n-1) - (points(n-1) - x0val)*coeffs_work(m,n-1,n-1))
      end do

      c1 = c2

    end do nloop

    coeffs = coeffs_work( deriv, size(points) - 1, : )

    deallocate(coeffs_work, stat = ierr(1) )
    if (ierr(1) .ne. 0) stop "Deallocation error in finite_d_coeffs_stencil"

  end subroutine finite_d_coeffs_stencil

  subroutine finite_d_func( func, points, step, deriv, stencil, res )
    real(kind = dp), external                                   :: func
    real(kind = dp), dimension(:),              intent( in    ) :: points
    real(kind = dp),                            intent( in    ) :: step
    integer,                                    intent( in    ) :: deriv
    integer,                                    intent( in    ) :: stencil
    real(kind = dp), dimension(:), allocatable, intent(   out ) :: res

    integer :: npoints
    real(kind = dp), dimension(:), allocatable :: stencil_points
    real(kind = dp), dimension(:), allocatable :: stencil_coeffs

    integer :: i,j
    integer :: ierr

    call finite_d_coeffs_stencil( stencil, deriv, 0, stencil_points, stencil_coeffs)

    npoints = size(points)

    if ( allocated (res) ) deallocate(res)

    allocate(res ( npoints ), stat = ierr )
    if (ierr .ne. 0) stop "Error allocating res in finite_d_func"

    res = 0.0_dp
    do i = 1, npoints

      do j = 0, 2*stencil
        res(i) = res(i) + func( points(i) + stencil_points(j)*step ) * stencil_coeffs(j)
      end do

    end do

    res = res / (step**deriv)

    deallocate(stencil_points, stat=ierr)
    if (ierr.ne.0) stop "Error deallocating stencil_points in finite_d_func"
    deallocate(stencil_coeffs, stat=ierr)
    if (ierr.ne.0) stop "Error deallocating stencil_coeffs in finite_d_func"


  end subroutine finite_d_func

  function finite_d_array_point( line, point, step, deriv, points, coeffs, boundary, res )
    real(kind = dp), dimension(:) :: line
    integer :: point
    real(kind = dp) :: step
    integer :: deriv
    real(kind = dp), dimension(:) :: coeffs
    real(kind = dp), optional     :: boundary
    real(kind = dp) :: res

    res = 0.0_dp
    npoints = size(points)
    nline   = size(line)
    
    if (present(boundary)) then
      do i = 1, 
        res = res + 
      end do
    else
      do i = 1

      end do
    end if



  end function finite_d_array_point


  subroutine finite_d_array_1d( grid, step, deriv, stencil, res, boundary )
    real(kind = dp), dimension(:),                              :: grid
    real(kind = dp),                            intent( in    ) :: step
    integer,                                    intent( in    ) :: deriv
    integer,                                    intent( in    ) :: stencil
    real(kind = dp), dimension(:), allocatable, intent(   out ) :: res
    real(kind = dp), optional                                   :: boundary

    integer :: npoints
    integer :: i
    integer :: x_max
    real(kind = dp), dimension(:), allocatable :: stencil_points
    real(kind = dp), dimension(:), allocatable :: stencil_coeffs

    x_max = size(grid)

    do i = 1, x_max



    end subroutine finite_d_array_1d

  end module finite_d


  program test
    use finite_d
    implicit none

    integer, parameter :: deriv = 1, stencil = 4, direction = 0
    real(kind = dp), dimension(:), allocatable :: points
    real(kind = dp), dimension(:), allocatable :: coeffs
    real(kind = dp), dimension(:), allocatable :: res

    call finite_d_coeffs_stencil( stencil, deriv, direction, points, coeffs)

    call finite_d_func  ( sqr, [1.0_dp], 0.4_dp, deriv, stencil, res )

    print*, res

  contains
    function sqr(x)
      real(kind=dp) :: sqr,x

      sqr = x**2

    end function sqr


  end program test
