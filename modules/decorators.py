import functools
import time
import logging

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def log_execution_time(logger_name=None):
    """
    Декоратор для логирования времени выполнения функций.
    
    Args:
        logger_name (str, optional): Имя логгера. Если не указано, используется имя модуля.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Получаем имя логгера
            if logger_name:
                logger = logging.getLogger(logger_name)
            else:
                logger = logging.getLogger(func.__module__)
            
            # Засекаем время начала выполнения
            start_time = time.time()
            
            # Выполняем функцию
            result = func(*args, **kwargs)
            
            # Вычисляем время выполнения
            execution_time = time.time() - start_time
            
            # Логируем результат
            logger.info(f"Function '{func.__name__}' executed in {execution_time:.2f} seconds")
            
            return result
        return wrapper
    return decorator 